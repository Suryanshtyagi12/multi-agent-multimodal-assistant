import json
import os
import sys
import time
import httpx
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DATASET_PATH = "evaluation/evaluation_dataset.json"
RESULTS_PATH = "evaluation/results.json"

SKIP_TEST_TYPES = [
    "empty_query",
    "prompt_injection"
]

def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        data = json.load(f)
    questions = data["questions"]
    valid = [
        q for q in questions
        if q.get("test_type") not in SKIP_TEST_TYPES
        and q.get("question", "").strip()
    ]
    logger.info(f"Loaded {len(valid)} valid questions from {len(questions)} total")
    return valid

def query_system(question: str) -> dict:
    try:
        response = httpx.post(
            f"{API_URL}/query",
            json={"query": question, "conversation_history": []},
            timeout=120
        )
        return response.json()
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {
            "answer": "",
            "route": "error",
            "sources": [],
            "reflection_note": str(e)
        }

def check_guardrail_behavior(question: dict, response: dict) -> dict:
    expected_blocked = question.get("test_type") in [
        "out_of_scope", "prompt_injection", "empty_query"
    ]
    was_blocked = response.get("route") == "guardrail_blocked"
    
    if expected_blocked:
        return {
            "guardrail_correct": was_blocked,
            "note": "correctly blocked" if was_blocked else "should have been blocked"
        }
    return {
        "guardrail_correct": not was_blocked,
        "note": "correctly allowed" if not was_blocked else "incorrectly blocked"
    }

def score_answer_relevance(question: str, answer: str) -> float:
    if not answer or len(answer.strip()) < 10:
        return 0.0
    
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "system",
                    "content": """You are an evaluation judge for a RAG system.
Score how relevant the answer is to the question on a scale of 0.0 to 1.0:
1.0 = perfectly answers the question
0.7 = mostly answers with minor gaps
0.5 = partially answers
0.3 = barely addresses the question
0.0 = completely irrelevant or empty
Reply with ONLY a decimal number between 0.0 and 1.0"""
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nAnswer: {answer[:500]}"
                }
            ],
            max_tokens=10,
            temperature=0
        )
        raw_text = response.choices[0].message.content
        import re
        matches = re.findall(r'(1\.0|0\.\d+)', raw_text)
        if matches:
            return float(matches[-1])
        if re.search(r'\b1\b', raw_text): return 1.0
        if re.search(r'\b0\b', raw_text): return 0.0
        return 0.5
    except Exception as e:
        logger.warning(f"Answer relevance scoring failed: {e}")
        return 0.5

def score_faithfulness(answer: str, contexts: list[str]) -> float:
    if not answer or not contexts:
        return 0.0
    
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    context_text = "\n---\n".join(contexts[:3])
    
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "system",
                    "content": """You are a faithfulness judge for a RAG system.
Check if the answer is grounded in the provided context.
Score 0.0 to 1.0:
1.0 = every claim in the answer is supported by the context
0.7 = most claims supported, minor additions
0.5 = half the claims supported
0.3 = few claims supported
0.0 = answer contradicts or ignores the context entirely
Reply with ONLY a decimal number between 0.0 and 1.0"""
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text[:1000]}\n\nAnswer: {answer[:500]}"
                }
            ],
            max_tokens=10,
            temperature=0
        )
        raw_text = response.choices[0].message.content
        import re
        matches = re.findall(r'(1\.0|0\.\d+)', raw_text)
        if matches:
            return float(matches[-1])
        if re.search(r'\b1\b', raw_text): return 1.0
        if re.search(r'\b0\b', raw_text): return 0.0
        return 0.5
    except Exception as e:
        logger.warning(f"Faithfulness scoring failed: {e}")
        return 0.5

def score_context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    if not contexts:
        return 0.0
    relevant = 0
    for ctx in contexts:
        key_words = ground_truth.lower().split()[:10]
        ctx_lower = ctx.lower()
        matches = sum(1 for w in key_words if w in ctx_lower and len(w) > 3)
        if matches >= 3:
            relevant += 1
    return round(relevant / len(contexts), 2)

def score_context_recall(ground_truth: str, contexts: list[str]) -> float:
    if not contexts or not ground_truth:
        return 0.0
    
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    context_text = "\n---\n".join(contexts[:5])
    
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "system",
                    "content": """You are a context recall judge for a RAG system.
Check if the retrieved context contains enough information 
to derive the ground truth answer.
Score 0.0 to 1.0:
1.0 = context contains all information needed for ground truth
0.7 = context contains most information
0.5 = context contains some information
0.3 = context contains little relevant information
0.0 = context is completely irrelevant to ground truth
Reply with ONLY a decimal number between 0.0 and 1.0"""
                },
                {
                    "role": "user",
                    "content": f"Ground truth: {ground_truth[:300]}\n\nContext:\n{context_text[:1000]}"
                }
            ],
            max_tokens=10,
            temperature=0
        )
        raw_text = response.choices[0].message.content
        import re
        matches = re.findall(r'(1\.0|0\.\d+)', raw_text)
        if matches:
            return float(matches[-1])
        if re.search(r'\b1\b', raw_text): return 1.0
        if re.search(r'\b0\b', raw_text): return 0.0
        return 0.5
    except Exception as e:
        logger.warning(f"Context recall scoring failed: {e}")
        return 0.5

def run_evaluation():
    print("=" * 50)
    print("ScholarRAG Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    health = httpx.get(f"{API_URL}/health", timeout=30).json()
    print(f"API Status: {health.get('status')}")
    print(f"Chunks indexed: {health.get('collection_stats', {}).get('total_chunks', 0)}")
    print()
    
    questions = load_dataset()
    results = []
    
    all_answer_relevance = []
    all_faithfulness = []
    all_context_precision = []
    all_context_recall = []
    
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_queries_measured = 0
    
    guardrail_correct = 0
    guardrail_total = 0
    
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['id']} — {q['question'][:60]}...")
        
        response = query_system(q["question"])
        
        answer = response.get("answer", "")
        sources = response.get("sources", [])
        route = response.get("route", "")
        contexts = [s.get("content", "") for s in sources if s.get("content")]
        
        latency = response.get("latency_seconds", 0.0)
        token_usage = response.get("token_usage", {})
        
        total_latency += latency
        if latency > 0:
            total_queries_measured += 1
            
        total_prompt_tokens += token_usage.get("prompt_tokens", 0)
        total_completion_tokens += token_usage.get("completion_tokens", 0)
        
        guardrail_result = check_guardrail_behavior(q, response)
        if q.get("test_type") in ["out_of_scope", "empty_query", "prompt_injection"]:
            guardrail_total += 1
            if guardrail_result["guardrail_correct"]:
                guardrail_correct += 1
        
        if q.get("test_type") in ["out_of_scope", "negative", "summarization_intent"]:
            result = {
                "id": q["id"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "answer": answer,
                "route": route,
                "context_type": q["context_type"],
                "test_type": q["test_type"],
                "difficulty": q["difficulty"],
                "source_paper": q["source_paper"],
                "guardrail_correct": guardrail_result["guardrail_correct"],
                "guardrail_note": guardrail_result["note"],
                "answer_relevance": None,
                "faithfulness": None,
                "context_precision": None,
                "context_recall": None,
                "num_sources": len(sources),
                "skipped_metrics": True
            }
            results.append(result)
            print(f"  Route: {route} | Guardrail: {guardrail_result['note']}")
            print()
            continue
        
        print(f"  Scoring metrics...")
        
        answer_relevance = score_answer_relevance(q["question"], answer)
        time.sleep(0.5)
        
        faithfulness = score_faithfulness(answer, contexts)
        time.sleep(0.5)
        
        context_precision = score_context_precision(
            q["question"], contexts, q["ground_truth"]
        )
        
        context_recall = score_context_recall(q["ground_truth"], contexts)
        time.sleep(0.5)
        
        all_answer_relevance.append(answer_relevance)
        all_faithfulness.append(faithfulness)
        all_context_precision.append(context_precision)
        all_context_recall.append(context_recall)
        
        result = {
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "answer": answer,
            "route": route,
            "context_type": q["context_type"],
            "test_type": q["test_type"],
            "difficulty": q["difficulty"],
            "source_paper": q["source_paper"],
            "guardrail_correct": guardrail_result["guardrail_correct"],
            "guardrail_note": guardrail_result["note"],
            "answer_relevance": answer_relevance,
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "num_sources": len(sources),
            "skipped_metrics": False
        }
        results.append(result)
        
        print(f"  Route: {route} | Sources: {len(sources)} | Latency: {latency:.2f}s")
        print(f"  Tokens: {token_usage.get('total_tokens', 0)} "
              f"(Prompt: {token_usage.get('prompt_tokens', 0)}, "
              f"Completion: {token_usage.get('completion_tokens', 0)})")
        print(f"  Answer Relevance: {answer_relevance:.2f} | "
              f"Faithfulness: {faithfulness:.2f} | "
              f"Precision: {context_precision:.2f} | "
              f"Recall: {context_recall:.2f}")
        print()
        
        time.sleep(1)
    
    avg_relevance = sum(all_answer_relevance) / len(all_answer_relevance) if all_answer_relevance else 0
    avg_faithfulness = sum(all_faithfulness) / len(all_faithfulness) if all_faithfulness else 0
    avg_precision = sum(all_context_precision) / len(all_context_precision) if all_context_precision else 0
    avg_recall = sum(all_context_recall) / len(all_context_recall) if all_context_recall else 0
    guardrail_accuracy = guardrail_correct / guardrail_total if guardrail_total > 0 else None
    avg_latency = total_latency / total_queries_measured if total_queries_measured > 0 else 0.0
    
    summary = {
        "evaluation_date": datetime.now().isoformat(),
        "model": "openai/gpt-oss-120b",
        "total_questions": len(questions),
        "questions_scored": len(all_answer_relevance),
        "questions_skipped": len(questions) - len(all_answer_relevance),
        "performance": {
            "average_latency_seconds": round(avg_latency, 3),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens
        },
        "metrics": {
            "answer_relevance": round(avg_relevance, 3),
            "faithfulness": round(avg_faithfulness, 3),
            "context_precision": round(avg_precision, 3),
            "context_recall": round(avg_recall, 3),
            "guardrail_accuracy": round(guardrail_accuracy, 3) if guardrail_accuracy else None
        },
        "results": results
    }
    
    os.makedirs("evaluation", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Questions scored: {len(all_answer_relevance)}/{len(questions)}")
    print()
    print("PERFORMANCE:")
    print(f"  Avg Latency/Query:  {avg_latency:.2f} seconds")
    print(f"  Total Tokens Used:  {total_prompt_tokens + total_completion_tokens:,}")
    print(f"  (Prompt: {total_prompt_tokens:,} | Completion: {total_completion_tokens:,})")
    print()
    print("METRICS:")
    print(f"  Answer Relevance:   {avg_relevance:.3f}")
    print(f"  Faithfulness:       {avg_faithfulness:.3f}")
    print(f"  Context Precision:  {avg_precision:.3f}")
    print(f"  Context Recall:     {avg_recall:.3f}")
    if guardrail_accuracy is not None:
        print(f"  Guardrail Accuracy: {guardrail_accuracy:.3f}")
    print()
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 50)
    
    return summary

if __name__ == "__main__":
    run_evaluation()
