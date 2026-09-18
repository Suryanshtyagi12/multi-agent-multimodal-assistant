import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from app.agents.graph_state import GraphState
from app.ingestion.embedder import embed_text
from app.retrieval.hybrid_retriever import hybrid_retrieve
from app.retrieval.chroma_store import get_collection, get_unique_sources_count
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scholarrag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
SIMILARITY_THRESHOLD = 0.4

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY,
    temperature=0
)

@tool
def search_text_chunks(query: str, n_results: int = 5) -> list:
    """Search methodology, abstract, introduction, discussion, 
    conclusions, citations, and author contributions from 
    research papers. Use for conceptual or explanatory questions."""
    logger.info(f"Tool called: search_text_chunks | query: {query[:50]}")
    return hybrid_retrieve(query=query, filter_type="text", n_results=n_results)

@tool
def search_table_chunks(query: str, n_results: int = 5) -> list:
    """Search results tables, metrics, accuracy scores, BLEU scores,
    F1 scores, benchmarks, and numerical comparisons from research 
    papers. Use for questions about specific numbers or rankings."""
    logger.info(f"Tool called: search_table_chunks | query: {query[:50]}")
    return hybrid_retrieve(query=query, filter_type="table", n_results=n_results)

@tool
def search_figure_chunks(query: str, n_results: int = 5) -> list:
    """Search figures, graphs, architecture diagrams, visualizations,
    plots, and charts from research papers. Use for questions about 
    visual content or model architecture diagrams."""
    logger.info(f"Tool called: search_figure_chunks | query: {query[:50]}")
    return hybrid_retrieve(query=query, filter_type="figure", n_results=n_results)

@tool
def search_all_chunks(query: str, n_results: int = 10) -> list:
    """Search across all content types — text, tables, and figures —
    when the query needs combined evidence or when the topic is 
    unclear. Use as fallback for ambiguous or complex questions."""
    logger.info(f"Tool called: search_all_chunks | query: {query[:50]}")
    return hybrid_retrieve(query=query, filter_type=None, n_results=n_results)

tools = [search_text_chunks, search_table_chunks, 
         search_figure_chunks, search_all_chunks]

llm_with_tools = llm.bind_tools(tools)

def embed_query_node(state: GraphState) -> dict:
    logger.info(f"Node: embed_query | query: {state['query'][:60]}")
    try:
        embedding = embed_text(state["query"])
        logger.info(f"embed_query: dimension={len(embedding)}")
        return {
            "query_embedding": embedding,
            "environment": ENVIRONMENT,
            "retry_count": 0,
            "reflection_note": "",
            "agent_outputs": {},
            "retrieved_chunks": [],
            "source_chunks": []
        }
    except Exception as e:
        logger.error(f"embed_query failed: {e}")
        return {"error": str(e)}

def tool_retrieval_node(state: GraphState) -> dict:
    logger.info(f"Node: tool_retrieval | query: {state['query'][:60]}")
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a retrieval agent for a research paper assistant.
                Given a user query, decide which search tools to call to find 
                the most relevant information. 
                
                Rules:
                - For questions about numbers, scores, metrics: call search_table_chunks
                - For questions about diagrams, figures, graphs: call search_figure_chunks  
                - For questions about methodology, concepts, text: call search_text_chunks
                - For ambiguous queries that could be in multiple places: 
                  call BOTH relevant tools (e.g. search_table_chunks AND search_text_chunks)
                - When completely uncertain: call search_all_chunks
                
                Always call at least one tool."""
            },
            {
                "role": "user", 
                "content": state["query"]
            }
        ]
        
        response = llm_with_tools.invoke(messages)
        logger.info(f"tool_retrieval: LLM chose {len(response.tool_calls)} tool(s)")
        
        num_tools_called = len(response.tool_calls)
        
        # Get unique paper count
        try:
            unique_papers = get_unique_sources_count()
        except:
            unique_papers = 1
        
        # Base n_results on both tool count AND paper count
        if num_tools_called > 1:
            # Multi-tool query needs more chunks per tool
            base_n = max(10, unique_papers * 2)
        else:
            # Single tool query — at least 1 chunk per paper
            base_n = max(5, unique_papers)
        
        # Hard cap at 15
        n_results = min(base_n, 15)
        
        logger.info(
            f"tool_retrieval: tools={num_tools_called} "
            f"papers={unique_papers} n_results={n_results}"
        )
        
        all_chunks = []
        seen_ids = set()
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = dict(tool_call["args"])
            
            tool_args["n_results"] = 10 if tool_name == "search_all_chunks" else n_results
            logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
            
            if tool_name == "search_text_chunks":
                chunks = search_text_chunks.invoke(tool_args)
            elif tool_name == "search_table_chunks":
                chunks = search_table_chunks.invoke(tool_args)
            elif tool_name == "search_figure_chunks":
                chunks = search_figure_chunks.invoke(tool_args)
            elif tool_name == "search_all_chunks":
                chunks = search_all_chunks.invoke(tool_args)
            else:
                chunks = []
            
            for chunk in chunks:
                if chunk.get("chunk_id") not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.get("chunk_id"))
                    if len(all_chunks) >= 15:
                        break
            if len(all_chunks) >= 15:
                break
        
        if not all_chunks:
            logger.warning("tool_retrieval: No chunks returned, falling back to search_all")
            all_chunks = search_all_chunks.invoke({"query": state["query"]})
        
        logger.info(f"tool_retrieval: {len(all_chunks)} unique chunks retrieved")
        return {"retrieved_chunks": all_chunks}
    
    except Exception as e:
        logger.error(f"tool_retrieval failed: {e}")
        return {"error": str(e), "retrieved_chunks": []}

def reflection_node(state: GraphState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)
    MAX_RETRIES = 2
    
    logger.info(f"Node: reflection | chunks={len(chunks)} retry={retry_count}")
    
    if retry_count >= MAX_RETRIES:
        note = f"Max retries ({MAX_RETRIES}) reached. Answering with best available."
        logger.warning(f"reflection: {note}")
        return {"reflection_note": note, "retry_count": retry_count}
    
    if not chunks:
        logger.warning("reflection: Empty chunks, retrying with search_all")
        new_chunks = search_all_chunks.invoke({"query": state["query"]})
        return {
            "retrieved_chunks": new_chunks,
            "reflection_note": f"Empty retrieval — retried broader search (attempt {retry_count + 1})",
            "retry_count": retry_count + 1
        }
    
    top_chunk = chunks[0]
    distance = top_chunk.get("distance", None)
    
    if distance is not None:
        similarity = 1 - distance
        if similarity < SIMILARITY_THRESHOLD:
            logger.warning(f"reflection: Low similarity {similarity:.2f}, retrying")
            new_chunks = search_all_chunks.invoke({"query": state["query"]})
            return {
                "retrieved_chunks": new_chunks,
                "reflection_note": f"Low similarity {similarity:.2f} — retried broader search (attempt {retry_count + 1})",
                "retry_count": retry_count + 1
            }
        note = f"Chunks passed similarity check: {similarity:.2f}"
    else:
        note = f"Chunks available: {len(chunks)} — proceeding"
    
    logger.info(f"reflection: {note}")
    return {"reflection_note": note, "retry_count": retry_count}

def answer_node(state: GraphState) -> dict:
    logger.info(f"Node: answer | chunks={len(state.get('retrieved_chunks', []))}")
    
    chunks = state.get("retrieved_chunks", [])
    
    if not chunks:
        return {
            "final_answer": "I could not find relevant information in the uploaded papers. Please try rephrasing your question or upload more relevant papers.",
            "source_chunks": []
        }
    
    context_parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        context_parts.append(
            f"Source: {meta.get('source_filename', 'unknown')} | "
            f"Page: {meta.get('page_number', '?')} | "
            f"Type: {meta.get('type', 'unknown')} | "
            f"Section: {meta.get('section_title', 'unknown')}\n"
            f"{chunk.get('content', '')}\n---"
        )
    context = "\n".join(context_parts)
    
    conversation_context = ""
    history = state.get("conversation_history", [])
    if history:
        recent = history[-4:]
        conversation_context = "\n".join([
            f"{m['role'].capitalize()}: {m['content']}" 
            for m in recent
        ]) + "\n\n"
    
    messages = [
        {
            "role": "system",
            "content": """You are a research paper assistant. Answer questions 
            using ONLY the provided context from academic papers.
            Always cite the source filename and page number for every claim.
            If the context does not contain enough information, say clearly:
            'The uploaded papers do not contain enough information to answer this.'
            Be precise, technical, and concise."""
        },
        {
            "role": "user",
            "content": f"{conversation_context}Context:\n{context}\n\nQuestion: {state['query']}"
        }
    ]
    
    response = llm.invoke(messages)
    answer = response.content
    token_usage = response.response_metadata.get('token_usage', {})
    
    if answer is None:
        answer = ""

    # Strip harmony/chain-of-thought reasoning blocks
    import re
    for tag in ["<think>", "<reasoning>", "<thought>", "<reflection>"]:
        end_tag = tag.replace("<", "</")
        pattern = f"{re.escape(tag)}.*?{re.escape(end_tag)}"
        answer = re.sub(pattern, "", answer, flags=re.DOTALL)

    answer = answer.strip()

    if not answer:
        answer = "I could not generate a response. Please try again."
    
    logger.info(f"answer_node: generated {len(answer)} char response")
    
    return {
        "final_answer": answer,
        "source_chunks": chunks,
        "agent_outputs": {"answer": answer, "token_usage": token_usage}
    }

def error_node(state: GraphState) -> dict:
    error = state.get("error", "Unknown error")
    logger.error(f"error_node reached: {error}")
    return {
        "final_answer": "I encountered an error processing your query. Please try again.",
        "source_chunks": [],
        "reflection_note": f"Error: {error}"
    }

def should_go_to_error(state: GraphState) -> str:
    if state.get("error"):
        return "error"
    return "continue"

graph = StateGraph(GraphState)

graph.add_node("embed_query", embed_query_node)
graph.add_node("tool_retrieval", tool_retrieval_node)
graph.add_node("reflection", reflection_node)
graph.add_node("answer", answer_node)
graph.add_node("error_handler", error_node)

graph.set_entry_point("embed_query")

graph.add_conditional_edges(
    "embed_query",
    should_go_to_error,
    {"error": "error_handler", "continue": "tool_retrieval"}
)

graph.add_conditional_edges(
    "tool_retrieval",
    should_go_to_error,
    {"error": "error_handler", "continue": "reflection"}
)

graph.add_edge("reflection", "answer")
graph.add_edge("answer", END)
graph.add_edge("error_handler", END)

rag_graph = graph.compile()

def run_query(query: str, conversation_history: list[dict] = None) -> dict:
    logger.info(f"run_query called: {query[:60]}")
    
    initial_state = {
        "query": query,
        "route": "",
        "query_embedding": [],
        "retrieved_chunks": [],
        "agent_outputs": {},
        "final_answer": "",
        "source_chunks": [],
        "conversation_history": conversation_history or [],
        "error": None,
        "environment": ENVIRONMENT,
        "reflection_note": "",
        "retry_count": 0
    }
    
    try:
        result = rag_graph.invoke(initial_state)
        logger.info(f"run_query complete | reflection: {result.get('reflection_note', '')}")
        return {
            "answer": result["final_answer"],
            "route": "tool_calling",
            "environment": result.get("environment", ENVIRONMENT),
            "reflection_note": result.get("reflection_note", ""),
            "sources": result.get("source_chunks", []),
            "token_usage": result.get("agent_outputs", {}).get("token_usage", {})
        }
    except Exception as e:
        logger.error(f"run_query failed: {e}")
        return {
            "answer": "Sorry, an error occurred. Please try again.",
            "route": "error",
            "environment": ENVIRONMENT,
            "reflection_note": f"Error: {str(e)}",
            "sources": [],
            "token_usage": {}
        }

if __name__ == "__main__":
    print("=== Agent Graph Test ===")
    print("NOTE: Requires ChromaDB to have ingested papers.")
    print("If ChromaDB is empty, tool calls will return empty results.")
    print()
    
    print("Test 1 — Graph compiles correctly:")
    print(f"  Nodes: {list(graph.nodes.keys())}")
    print(f"  Tools registered: {[t.name for t in tools]}")
    print()
    
    print("Test 2 — run_query with empty ChromaDB:")
    result = run_query("What is the attention mechanism?")
    print(f"  Answer: {result['answer'][:100]}")
    print(f"  Route: {result['route']}")
    print(f"  Reflection: {result['reflection_note']}")
    print(f"  Sources: {len(result['sources'])}")
    print()
    
    print("Test 3 — Check LangSmith tracing:")
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key and langsmith_key != "your_langsmith_key_here":
        print("  LangSmith: configured ✓")
        print("  Check traces at: smith.langchain.com")
    else:
        print("  LangSmith: not configured — add LANGCHAIN_API_KEY to .env")
    
    print()
    print("Agent graph ready.")
