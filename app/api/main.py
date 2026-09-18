import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import torch
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import shutil
import time
from dotenv import load_dotenv
load_dotenv()

from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.chunker import chunk_elements
from app.ingestion.figure_captioner import caption_all_figures
from app.ingestion.embedder import embed_batch
from app.retrieval.chroma_store import (
    add_chunks, delete_collection, get_collection_stats
)
from app.retrieval.bm25_index import bm25_index
from app.agents.agent_graph import run_query
from app.guardrails.guardrails import (
    run_input_guardrails,
    run_output_guardrails
)

# Force root logger to write to file
file_handler = logging.FileHandler('scholarrag.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

app = FastAPI(
    title="ScholarRAG API",
    description="Multimodal RAG system for research papers",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[list] = []

@app.on_event("startup")
async def startup_event():
    logger.info(f"ScholarRAG API starting — environment: {ENVIRONMENT}")
    print(f"ScholarRAG API ready — environment: {ENVIRONMENT}")

@app.post("/ingest")
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                400,
                f"{file.filename} is not a PDF. Only PDF files supported."
            )
    
    all_chunks = []
    results_per_file = []
    temp_paths = []
    
    try:
        for file in files:
            logger.info(f"/ingest: processing {file.filename}")
            print(f"\nProcessing: {file.filename}")
            
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f"_{file.filename}"
            ) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name
                temp_paths.append(temp_path)
            
            try:
                print(f"  Parsing {file.filename}...")
                elements = parse_pdf(temp_path)
                
                if not elements:
                    logger.warning(f"No elements from {file.filename}")
                    results_per_file.append({
                        "filename": file.filename,
                        "status": "skipped",
                        "reason": "No extractable content found",
                        "total_chunks": 0
                    })
                    continue
                
                print(f"  Chunking {file.filename}...")
                chunks = chunk_elements(elements)
                
                if not chunks:
                    results_per_file.append({
                        "filename": file.filename,
                        "status": "skipped",
                        "reason": "No chunks created",
                        "total_chunks": 0
                    })
                    continue
                
                print(f"  Captioning figures in {file.filename}...")
                chunks = caption_all_figures(chunks)
                
                all_chunks.extend(chunks)
                
                text_count = sum(1 for c in chunks if c["type"] == "text")
                table_count = sum(1 for c in chunks if c["type"] == "table")
                figure_count = sum(1 for c in chunks if c["type"] == "figure")
                
                results_per_file.append({
                    "filename": file.filename,
                    "status": "success",
                    "total_chunks": len(chunks),
                    "text_chunks": text_count,
                    "table_chunks": table_count,
                    "figure_chunks": figure_count
                })
                
                logger.info(
                    f"{file.filename}: {len(chunks)} chunks — "
                    f"text={text_count} table={table_count} "
                    f"figure={figure_count}"
                )
            
            except Exception as e:
                logger.error(f"Failed {file.filename}: {e}")
                results_per_file.append({
                    "filename": file.filename,
                    "status": "error",
                    "reason": str(e),
                    "total_chunks": 0
                })
                continue
        
        if not all_chunks:
            return {
                "status": "warning",
                "message": "No chunks extracted from any uploaded file",
                "total_chunks": 0,
                "files": results_per_file,
                "environment": ENVIRONMENT
            }
        
        print(f"\nEmbedding {len(all_chunks)} total chunks...")
        texts = [c["content"] for c in all_chunks]
        embeddings = embed_batch(texts)
        
        print("Storing all chunks in ChromaDB...")
        add_chunks(all_chunks, embeddings)
        
        print("Building BM25 index over all chunks...")
        bm25_index.build(all_chunks)
        
        successful = [r for r in results_per_file if r["status"] == "success"]
        
        logger.info(
            f"/ingest complete: {len(successful)}/{len(files)} files | "
            f"total={len(all_chunks)} chunks"
        )
        
        return {
            "status": "success",
            "total_files_processed": len(successful),
            "total_files_failed": len(files) - len(successful),
            "total_chunks": len(all_chunks),
            "files": results_per_file,
            "environment": ENVIRONMENT
        }
    
    except Exception as e:
        logger.error(f"/ingest failed: {e}")
        raise HTTPException(500, detail=str(e))
    
    finally:
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.post("/query")
async def query_papers(request: QueryRequest):
    start_time = time.time()
    
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    
    logger.info(f"/query: {request.query[:60]}")
    
    # Check if documents exist
    stats = get_collection_stats()
    has_documents = stats.get("total_chunks", 0) > 0
    
    # Run input guardrails
    guardrail_result = run_input_guardrails(
        request.query, 
        has_documents
    )
    
    if not guardrail_result["allowed"]:
        logger.warning(f"Query blocked by guardrails: {request.query[:50]}")
        return {
            "answer": guardrail_result["message"],
            "route": "guardrail_blocked",
            "environment": ENVIRONMENT,
            "reflection_note": "Blocked by input guardrails",
            "sources": [],
            "warning": None
        }
    
    try:
        result = run_query(
            query=request.query,
            conversation_history=request.conversation_history
        )
        
        # Run output guardrails
        output_check = run_output_guardrails(
            result["answer"],
            result.get("sources", [])
        )
        
        final_answer = output_check["answer"]
        warning = output_check.get("warning_message")
        
        if warning:
            final_answer = f"{final_answer}\n\n{warning}"
        
        formatted_sources = []
        for s in result.get("sources", []):
            meta = s.get("metadata", {})
            formatted_sources.append({
                "content": s.get("content", "")[:300],
                "source_filename": meta.get("source_filename", ""),
                "page_number": meta.get("page_number", 0),
                "type": meta.get("type", ""),
                "section_title": meta.get("section_title", "")
            })
        
        logger.info(
            f"/query complete | route={result.get('route')} | "
            f"warning={output_check.get('warning')}"
        )
        
        latency_seconds = time.time() - start_time
        
        return {
            "answer": final_answer,
            "route": result.get("route", ""),
            "environment": result.get("environment", ENVIRONMENT),
            "reflection_note": result.get("reflection_note", ""),
            "sources": formatted_sources,
            "warning": output_check.get("warning"),
            "latency_seconds": latency_seconds,
            "token_usage": result.get("token_usage", {})
        }
    
    except Exception as e:
        logger.error(f"/query failed: {e}")
        return {
            "answer": "Sorry, an error occurred. Please try again.",
            "route": "error",
            "environment": ENVIRONMENT,
            "reflection_note": f"Error: {str(e)}",
            "sources": [],
            "warning": None,
            "latency_seconds": time.time() - start_time,
            "token_usage": {}
        }

@app.delete("/collection")
async def clear_collection():
    try:
        delete_collection()
        bm25_index.reset()
        logger.info("/collection: cleared successfully")
        return {
            "status": "success",
            "message": f"Collection cleared for environment: {ENVIRONMENT}"
        }
    except Exception as e:
        logger.error(f"/collection delete failed: {e}")
        raise HTTPException(500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        stats = get_collection_stats()
        bm25_stats = bm25_index.get_stats()
        return {
            "status": "ok",
            "environment": ENVIRONMENT,
            "collection_stats": stats,
            "bm25_stats": bm25_stats
        }
    except Exception as e:
        logger.error(f"/health failed: {e}")
        return {
            "status": "degraded",
            "environment": ENVIRONMENT,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
