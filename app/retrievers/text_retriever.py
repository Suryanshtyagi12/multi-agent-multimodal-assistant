# app/retrievers/text_retriever.py

import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.Client()
collection = client.get_or_create_collection("text_docs")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_text(query, k=5):
    query_emb = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k
    )

    return results
