# app/retrievers/text_retriever.py

import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_PATH

# ---------------------------
# Embedding Model (MUST MATCH INGESTION)
# ---------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


# ---------------------------
# Text Retrieval
# ---------------------------
def retrieve_text(query: str, k: int = 5):
    """
    Retrieve top-k text chunks from ChromaDB
    """

    if not query or not query.strip():
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # ---------------------------
    # Load ChromaDB
    # ---------------------------
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="text_docs"
    )

    doc_count = collection.count()
    print(f"🔍 [TEXT RETRIEVER] Chroma path: {CHROMA_PATH}")
    print(f"📦 [TEXT RETRIEVER] Collection count: {doc_count}")

    if doc_count == 0:
        print("❌ [TEXT RETRIEVER] No documents found")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # ---------------------------
    # Embed Query
    # ---------------------------
    query_embedding = embed_model.encode(query).tolist()

    # ---------------------------
    # Query Chroma
    # ---------------------------
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    print(f"✅ [TEXT RETRIEVER] Retrieved {len(results['documents'][0])} chunks")

    return results
