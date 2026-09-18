import os
import chromadb

EMBEDDING_CONFIG = {
    "local": {"dims": 1024, "collection": "research_papers_local"},
    "production": {"dims": 768, "collection": "research_papers_prod"}
}

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
if ENVIRONMENT not in EMBEDDING_CONFIG:
    ENVIRONMENT = "local"
    
COLLECTION_NAME = EMBEDDING_CONFIG[ENVIRONMENT]["collection"]
EMBEDDING_DIMS = EMBEDDING_CONFIG[ENVIRONMENT]["dims"]

# Initialize PersistentClient
client = chromadb.PersistentClient(path="./chroma_db")

def get_collection():
    """Returns the correct collection based on ENVIRONMENT, creates it if it does not exist."""
    return client.get_or_create_collection(name=COLLECTION_NAME)

def add_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """Adds chunks to the collection after validating dimension match."""
    if not chunks or not embeddings:
        return
        
    if len(embeddings[0]) != EMBEDDING_DIMS:
        raise ValueError(f"Dimension mismatch for environment '{ENVIRONMENT}'. Expected {EMBEDDING_DIMS} dims, but got {len(embeddings[0])}. Check your embedding model.")
        
    collection = get_collection()
    
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["content"])
        
        metadata = {
            "type": chunk["type"],
            "source_filename": chunk["source_filename"],
            "page_number": chunk.get("page_number", 0),
            "section_title": chunk.get("section_title", ""),
            "image_path": chunk.get("image_path") or "",
            "chunking_method": chunk.get("chunking_method", "unknown")
        }
        metadatas.append(metadata)
        
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

def query_dense(query_embedding: list[float], n_results: int = 20, filter_type: str = None) -> list[dict]:
    """Queries the collection by dense embedding, optionally filtering by type."""
    collection = get_collection()
    
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["metadatas", "documents", "distances"]
    }
    
    if filter_type:
        query_kwargs["where"] = {"type": filter_type}
        
    results = collection.query(**query_kwargs)
    
    parsed_results = []
    if results and results.get('ids') and len(results['ids']) > 0:
        for i in range(len(results['ids'][0])):
            parsed_results.append({
                "chunk_id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
            
    return parsed_results

def delete_collection():
    """Deletes the current environment's collection only."""
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    get_collection()

def get_collection_stats() -> dict:
    """Returns statistics about the current collection."""
    collection = get_collection()
    count = collection.count()
    
    text_chunks = 0
    table_chunks = 0
    figure_chunks = 0
    
    if count > 0:
        try:
            text_res = collection.get(where={"type": "text"}, include=[])
            text_chunks = len(text_res['ids']) if text_res and text_res.get('ids') else 0
            
            table_res = collection.get(where={"type": "table"}, include=[])
            table_chunks = len(table_res['ids']) if table_res and table_res.get('ids') else 0
            
            figure_res = collection.get(where={"type": "figure"}, include=[])
            figure_chunks = len(figure_res['ids']) if figure_res and figure_res.get('ids') else 0
        except Exception:
            pass

    unique_sources = get_unique_sources_count()

    return {
        "environment": ENVIRONMENT,
        "collection_name": COLLECTION_NAME,
        "embedding_dims": EMBEDDING_DIMS,
        "total_chunks": count,
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "figure_chunks": figure_chunks,
        "unique_sources": unique_sources
    }

def get_unique_sources_count() -> int:
    try:
        collection = get_collection()
        if collection.count() == 0:
            return 0
        result = collection.get(include=["metadatas"])
        filenames = set()
        for meta in result["metadatas"]:
            if meta and meta.get("source_filename"):
                filenames.add(meta["source_filename"])
        return len(filenames)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"get_unique_sources_count failed: {e}")
        return 1
