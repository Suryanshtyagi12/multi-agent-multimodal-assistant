# app/retrievers/image_retriever.py

import chromadb
from sentence_transformers import SentenceTransformer

# Load same CLIP model
clip_model = SentenceTransformer("clip-ViT-B-32")

# Same collection
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection("image_docs")

def retrieve_images(query, k=5):
    """Find relevant images for a text query."""
    query_emb = clip_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k
    )

    return results
