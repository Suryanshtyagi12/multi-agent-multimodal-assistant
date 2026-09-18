import sys
import os
import logging

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.ingestion.embedding_manager import embed_text as _embed_text
from app.ingestion.embedding_manager import embed_batch as _embed_batch

logger = logging.getLogger(__name__)

def embed_text(text: str, is_query: bool = False) -> list[float]:
    """
    Generates an embedding for a single text string based on the environment.
    Delegates to embedding_manager to handle BGE-M3 (local) or Gemini (production).
    """
    return _embed_text(text, is_query=is_query)

def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Generates embeddings for a batch of text strings based on the environment.
    Delegates to embedding_manager to handle BGE-M3 (local) or Gemini (production).
    """
    return _embed_batch(texts, is_query=is_query)

if __name__ == "__main__":
    vec = embed_text("What is the methodology used in this paper?", is_query=True)
    if vec:
        print(f"Embedding dimension: {len(vec)}")
        print(f"First 5 values: {vec[:5]}")
