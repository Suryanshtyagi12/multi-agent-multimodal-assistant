import os
import re
import logging
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def strip_reasoning(answer: str) -> str:
    if answer is None:
        answer = ""
    for tag in ["<think>", "<reasoning>", "<thought>", "<reflection>"]:
        end_tag = tag.replace("<", "</")
        pattern = f"{re.escape(tag)}.*?{re.escape(end_tag)}"
        answer = re.sub(pattern, "", answer, flags=re.DOTALL)
    return answer.strip()

# ====================================================================
# GUARDRAIL 1 — Input validation
# ====================================================================
def validate_input(query: str) -> dict:
    
    # Check 1 — empty or too short:
    if not query or len(query.strip()) < 3:
        return {
            "allowed": False,
            "reason": "Query too short",
            "message": "Please ask a complete question."
        }
    
    # Check 2 — too long:
    if len(query) > 1000:
        return {
            "allowed": False,
            "reason": "Query too long",
            "message": "Please keep your question under 1000 characters."
        }
    
    # Check 3 — clearly off-topic using keyword list:
    off_topic_keywords = [
        "recipe", "cook", "weather", "sports", "movie",
        "music", "celebrity", "news", "stock price",
        "lottery", "joke", "poem", "song"
    ]
    query_lower = query.lower()
    for keyword in off_topic_keywords:
        if keyword in query_lower:
            return {
                "allowed": False,
                "reason": "Off-topic query",
                "message": "I can only answer questions about uploaded research papers. This query appears unrelated to academic research."
            }
    
    # Check 4 — harmful content using LLM:
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": """You are a content safety classifier.
                    Classify if this query is safe and appropriate for 
                    an academic research assistant.
                    Reply with ONLY one word: SAFE or UNSAFE.
                    UNSAFE means: harmful, illegal, explicit, or 
                    completely unrelated to academic research."""
                },
                {"role": "user", "content": query}
            ],
            max_tokens=5,
            temperature=0
        )
        verdict = strip_reasoning(response.choices[0].message.content).upper()
        if verdict == "UNSAFE":
            return {
                "allowed": False,
                "reason": "Content safety check failed",
                "message": "This query cannot be processed. Please ask questions related to your research papers."
            }
    except Exception as e:
        logger.warning(f"Safety check failed, allowing query: {e}")
    
    return {"allowed": True, "reason": "passed", "message": ""}

# ====================================================================
# GUARDRAIL 2 — Out of scope detection
# ====================================================================
def check_scope(query: str, has_documents: bool) -> dict:
    
    if not has_documents:
        return {
            "in_scope": False,
            "message": "No research papers are uploaded yet. Please upload a PDF first using the sidebar."
        }
    
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": """You are a scope classifier for a 
                    research paper assistant.
                    Determine if this query is asking about academic 
                    research content — methodology, results, figures,
                    tables, citations, comparisons between papers,
                    or summaries of papers.
                    Reply with ONLY one word: IN_SCOPE or OUT_OF_SCOPE"""
                },
                {"role": "user", "content": query}
            ],
            max_tokens=10,
            temperature=0
        )
        verdict = strip_reasoning(response.choices[0].message.content).upper()
        
        if "OUT_OF_SCOPE" in verdict:
            return {
                "in_scope": False,
                "message": "This question appears to be outside the scope of your uploaded research papers. Try asking about methodology, results, figures, or comparisons between your papers."
            }
    except Exception as e:
        logger.warning(f"Scope check failed, allowing: {e}")
    
    return {"in_scope": True, "message": ""}

# ====================================================================
# GUARDRAIL 3 — Output validation
# ====================================================================
def validate_output(answer: str, context_chunks: list[dict]) -> dict:
    
    # Check 1 — empty answer:
    if not answer or len(answer.strip()) < 10:
        return {
            "valid": False,
            "answer": "I could not generate a response. Please try rephrasing your question.",
            "warning": "empty_answer"
        }
    
    # Check 2 — answer too long (LLM rambling):
    if len(answer) > 3000:
        answer = answer[:3000] + "...\n\n[Response truncated for clarity]"
    
    # Check 3 — hallucination check:
    if context_chunks:
        context_text = " ".join([
            c.get("content", "")[:200] for c in context_chunks[:3]
        ])
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a faithfulness checker.
                        Check if the answer is grounded in the provided 
                        context or if it contains hallucinated information
                        not present in the context.
                        Reply with ONLY one word: FAITHFUL or HALLUCINATED"""
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context_text}\n\nAnswer: {answer[:500]}"
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            verdict = strip_reasoning(response.choices[0].message.content).upper()
            
            if "HALLUCINATED" in verdict:
                logger.warning("Hallucination detected in answer")
                return {
                    "valid": True,
                    "answer": answer,
                    "warning": "possible_hallucination",
                    "warning_message": "⚠️ Note: This answer may contain information not directly from your uploaded papers. Please verify with the source chunks shown below."
                }
        except Exception as e:
            logger.warning(f"Hallucination check failed: {e}")
    
    return {"valid": True, "answer": answer, "warning": None}

# ====================================================================
# GUARDRAIL 4 — Summarization detector
# ====================================================================
def detect_summarization_intent(query: str) -> dict:
    summarize_keywords = [
        "summarize", "summary", "overview", "brief", 
        "explain the paper", "what is this paper about",
        "give me an overview", "tldr", "tl;dr",
        "main points", "key points", "outline"
    ]
    query_lower = query.lower()
    for keyword in summarize_keywords:
        if keyword in query_lower:
            specific_paper = None
            return {
                "is_summarization": True,
                "specific_paper": specific_paper
            }
    return {"is_summarization": False, "specific_paper": None}

# ====================================================================
# MAIN GUARDRAIL RUNNER
# ====================================================================
def run_input_guardrails(query: str, has_documents: bool) -> dict:
    
    # step 1 — validate input:
    input_check = validate_input(query)
    if not input_check["allowed"]:
        logger.warning(f"Input blocked: {input_check['reason']}")
        return {
            "allowed": False,
            "message": input_check["message"],
            "is_summarization": False
        }
    
    # step 2 — check scope:
    scope_check = check_scope(query, has_documents)
    if not scope_check["in_scope"]:
        logger.warning("Query out of scope")
        return {
            "allowed": False,
            "message": scope_check["message"],
            "is_summarization": False
        }
    
    # step 3 — detect summarization:
    summary_check = detect_summarization_intent(query)
    
    return {
        "allowed": True,
        "message": "",
        "is_summarization": summary_check["is_summarization"],
        "specific_paper": summary_check.get("specific_paper")
    }

def run_output_guardrails(
    answer: str, 
    context_chunks: list[dict]
) -> dict:
    return validate_output(answer, context_chunks)
