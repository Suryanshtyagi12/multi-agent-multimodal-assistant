import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PREFIX = "Represent this sentence for searching relevant passages: "
LOCAL_MODEL_ID = "BAAI/bge-m3"

# Determine environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "local").lower()

# Global state for lazy loading
_gemini_configured = False

def _init_gemini():
    """Initializes the Gemini API for production embedding."""
    global _gemini_configured
    if not _gemini_configured:
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY is missing but ENVIRONMENT=production.")
                raise ValueError("Missing GEMINI_API_KEY for production embeddings.")
            genai.configure(api_key=api_key)
            _gemini_configured = True
        except ImportError:
            logger.error("Failed to import google.generativeai. Did you install it?")
            raise
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            raise

def embed_text(text: str, is_query: bool = False) -> list[float]:
    """
    Generates an embedding for a single text string based on the environment.
    """
    if not text:
        return []

    try:
        if ENVIRONMENT == "production":
            _init_gemini()
            import google.generativeai as genai
            task_type = "retrieval_query" if is_query else "retrieval_document"
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type=task_type,
            )
            return result['embedding']
        else:
            global _local_model
            if '_local_model' not in globals():
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local SentenceTransformer model: {LOCAL_MODEL_ID} (This may take a moment to download)")
                _local_model = SentenceTransformer(LOCAL_MODEL_ID)
            
            prefix_text = PREFIX + text if is_query else text
            result = _local_model.encode([prefix_text], normalize_embeddings=True)
            return result[0].tolist()
            
    except Exception as e:
        logger.error(f"Failed to embed text in {ENVIRONMENT} environment: {e}")
        return []

def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Generates embeddings for a batch of text strings based on the environment.
    """
    if not texts:
        return []

    try:
        if ENVIRONMENT == "production":
            _init_gemini()
            import google.generativeai as genai
            task_type = "retrieval_query" if is_query else "retrieval_document"
            # Gemini embed_content accepts a list of strings
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=texts,
                task_type=task_type,
            )
            return result['embedding']
        else:
            global _local_model
            if '_local_model' not in globals():
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local SentenceTransformer model: {LOCAL_MODEL_ID} (This may take a moment to download)")
                _local_model = SentenceTransformer(LOCAL_MODEL_ID)
                
            prefixed_texts = [PREFIX + t if is_query else t for t in texts]
            results = _local_model.encode(prefixed_texts, normalize_embeddings=True)
            return results.tolist()
            
    except Exception as e:
        logger.error(f"Failed to embed batch in {ENVIRONMENT} environment: {e}")
        return [[] for _ in texts]

if __name__ == "__main__":
    print(f"--- Testing {ENVIRONMENT} Embedding ---")
    query_vec = embed_text("What is the methodology?", is_query=True)
    if query_vec:
        print(f"Dimension: {len(query_vec)}")
        print(f"First 5 values: {query_vec[:5]}")
    else:
        print("Failed to get embedding.")
