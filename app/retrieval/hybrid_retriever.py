import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
import logging
from sentence_transformers import CrossEncoder
from app.retrieval.chroma_store import query_dense, get_unique_sources_count
from app.retrieval.bm25_index import bm25_index
from app.ingestion.embedder import embed_text

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize cross encoder globally so it's only loaded once
try:
    logger.info("Loading local CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    logger.error(f"Failed to load CrossEncoder: {e}")
    cross_encoder = None

RRF_K = 60
SIMILARITY_THRESHOLD = 0.4

def reciprocal_rank_fusion(
    dense_results: list[dict], 
    sparse_results: list[dict], 
    k: int = RRF_K
) -> list[dict]:
    scores = {}
    chunk_data = {}
    
    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1/(k + rank)
        chunk_data[cid] = chunk
        chunk_data[cid]["retrieval_method"] = "dense"
    
    for rank, chunk in enumerate(sparse_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1/(k + rank)
        if cid not in chunk_data:
            chunk_data[cid] = chunk
        chunk_data[cid]["retrieval_method"] = "both" if chunk_data[cid].get("retrieval_method") == "dense" else "sparse"
    
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    results = []
    for cid in sorted_ids:
        chunk = chunk_data[cid]
        chunk["rrf_score"] = scores[cid]
        results.append(chunk)
    
    return results

def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    if not chunks:
        return []
    
    if cross_encoder is None:
        print("Reranker: CrossEncoder not loaded, skipping reranking")
        return chunks[:top_n]
    
    try:
        pairs = [[query, chunk["content"]] for chunk in chunks]
        scores = cross_encoder.predict(pairs)
        
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk["reranker_score"] = float(scores[i])
            scored_chunks.append(chunk)
        
        scored_chunks.sort(key=lambda x: x["reranker_score"], reverse=True)
        return scored_chunks[:top_n]
    
    except Exception as e:
        print(f"Reranker unavailable: {e}. Falling back to RRF order.")
        return chunks[:top_n]

def hybrid_retrieve(
    query: str,
    query_embedding: list[float] = None,
    n_results: int = 5,
    filter_type: str = None
) -> list[dict]:
    
    if query_embedding is None:
        query_embedding = embed_text(query)
    
    # Get unique paper count for dynamic scaling
    try:
        unique_papers = get_unique_sources_count()
    except:
        unique_papers = 1
    
    # Calculate final chunks to send to LLM
    # At least 1 chunk per paper, minimum 5, maximum 15
    final_n = max(n_results, unique_papers)
    final_n = min(final_n, 15)
    
    # Scale initial retrieval proportionally so reranker has enough candidates
    # Fetch at least 3x final_n from each index
    fetch_n = max(20, final_n * 3)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"hybrid_retrieve: unique_papers={unique_papers} "
        f"final_n={final_n} fetch_n={fetch_n} filter={filter_type}"
    )
    
    dense = query_dense(
        query_embedding=query_embedding,
        n_results=fetch_n,
        filter_type=filter_type
    )
    
    sparse = bm25_index.query(
        query_text=query,
        n_results=fetch_n,
        filter_type=filter_type
    )
    
    print(f"Retrieval: dense={len(dense)} sparse={len(sparse)} "
          f"filter={filter_type} final_n={final_n}")
    
    if not dense and not sparse:
        return []
    if not dense:
        return sparse[:final_n]
    if not sparse:
        return dense[:final_n]
    
    fused = reciprocal_rank_fusion(dense, sparse)
    print(f"RRF fusion: {len(fused)} unique chunks")
    
    # Pass fetch_n candidates to reranker, cut to final_n
    reranked = rerank(query, fused[:fetch_n], top_n=final_n)
    print(f"After reranking: {len(reranked)} final chunks "
          f"(dynamic based on {unique_papers} papers)")
    
    return reranked

if __name__ == "__main__":
    print("=== Hybrid Retriever Test ===")
    print("Testing RRF fusion only (no ChromaDB needed)")
    
    fake_dense = [
        {"chunk_id": "chunk_a", "content": "attention mechanism transformer", 
         "metadata": {"type": "text"}, "distance": 0.2},
        {"chunk_id": "chunk_b", "content": "BLEU score results table",
         "metadata": {"type": "table"}, "distance": 0.4},
        {"chunk_id": "chunk_c", "content": "encoder decoder diagram",
         "metadata": {"type": "figure"}, "distance": 0.5},
    ]
    
    fake_sparse = [
        {"chunk_id": "chunk_b", "content": "BLEU score results table",
         "metadata": {"type": "table"}, "score": 1.2},
        {"chunk_id": "chunk_a", "content": "attention mechanism transformer",
         "metadata": {"type": "text"}, "score": 0.8},
        {"chunk_id": "chunk_d", "content": "training data preprocessing",
         "metadata": {"type": "text"}, "score": 0.3},
    ]
    
    print("\nTest 1 — RRF fusion:")
    fused = reciprocal_rank_fusion(fake_dense, fake_sparse)
    for r in fused:
        print(f"  {r['chunk_id']} | rrf_score: {r['rrf_score']:.4f} | method: {r['retrieval_method']}")
    
    print("\nExpected: chunk_a and chunk_b highest (appear in both lists)")
    print("Expected: chunk_d lowest (sparse only)")
    
    print("\nTest 2 — RRF correctness check:")
    top = fused[0]
    assert top["chunk_id"] in ["chunk_a", "chunk_b"], "Top chunk should be in both lists"
    print(f"  Top chunk: {top['chunk_id']} — PASSED")
    
    print("\nHybrid retriever module loaded successfully")
    print("Full pipeline test: run via FastAPI /query endpoint after Prompt 10")
