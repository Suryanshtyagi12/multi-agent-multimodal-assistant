import re
from rank_bm25 import BM25Okapi

MIN_TOKEN_LENGTH = 2

def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return [t for t in tokens if len(t) >= MIN_TOKEN_LENGTH]

class BM25Index:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.is_built = False
        self.total_chunks = 0

    def build(self, chunks: list[dict]):
        if not chunks:
            self.is_built = False
            print("BM25: No chunks provided, index not built")
            return
        
        tokenized_corpus = [_tokenize(chunk.get("content", "")) for chunk in chunks]
        self.index = BM25Okapi(tokenized_corpus)
        self.chunks = chunks
        self.is_built = True
        self.total_chunks = len(chunks)
        print(f"BM25 index built: {len(chunks)} chunks indexed")

    def query(self, query_text: str, n_results: int = 20, filter_type: str = None) -> list[dict]:
        if not self.is_built:
            print("BM25: Index not built yet, returning empty")
            return []
        
        tokens = _tokenize(query_text)
        if not tokens:
            return []
        
        scores = self.index.get_scores(tokens)
        
        if filter_type is not None:
            for i, chunk in enumerate(self.chunks):
                if chunk.get("type") != filter_type:
                    scores[i] = 0.0
        
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:n_results]
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "metadata": {
                    "type": chunk.get("type"),
                    "source_filename": chunk.get("source_filename"),
                    "page_number": chunk.get("page_number"),
                    "section_title": chunk.get("section_title"),
                    "image_path": chunk.get("image_path", "")
                },
                "score": float(scores[idx]),
                "retrieval_method": "bm25"
            })
        
        return results

    def reset(self):
        self.index = None
        self.chunks = []
        self.is_built = False
        self.total_chunks = 0
        print("BM25 index reset")

    def get_stats(self) -> dict:
        return {
            "is_built": self.is_built,
            "total_chunks": self.total_chunks,
            "index_type": "BM25Okapi"
        }

bm25_index = BM25Index()

if __name__ == "__main__":
    test_chunks = [
        {
            "chunk_id": "test_1",
            "type": "text",
            "content": "transformer architecture attention mechanism self attention multi head",
            "source_filename": "attention.pdf",
            "page_number": 1,
            "section_title": "Introduction",
            "image_path": None
        },
        {
            "chunk_id": "test_2",
            "type": "table",
            "content": "BLEU score WMT14 English German translation benchmark results comparison",
            "source_filename": "attention.pdf",
            "page_number": 6,
            "section_title": "Results",
            "image_path": None
        },
        {
            "chunk_id": "test_3",
            "type": "figure",
            "content": "encoder decoder architecture diagram multi head attention layers",
            "source_filename": "attention.pdf",
            "page_number": 3,
            "section_title": "Model Architecture",
            "image_path": "figures/fig1.png"
        }
    ]
    
    print("=== BM25 Index Test ===")
    bm25_index.build(test_chunks)
    
    print("\nTest 1 — keyword search:")
    results = bm25_index.query("attention mechanism", n_results=3)
    for r in results:
        print(f"  {r['chunk_id']} | score: {r['score']:.3f} | {r['content'][:50]}")
    
    print("\nTest 2 — filtered search (table only):")
    results = bm25_index.query("BLEU score results", n_results=3, filter_type="table")
    for r in results:
        print(f"  {r['chunk_id']} | type: {r['metadata']['type']} | score: {r['score']:.3f}")
    
    print("\nTest 3 — query with no matches:")
    results = bm25_index.query("quantum physics string theory", n_results=3)
    print(f"  Results returned: {len(results)} (expected 0 or low scores)")
    
    print("\nStats:", bm25_index.get_stats())
    
    bm25_index.reset()
    print("Reset done:", bm25_index.get_stats())
