# Progress tracker
Last updated: 2026-06-03

## Session log
**Session 1 — Project setup**

- Deleted all test/debug files from root (debug_chroma.py, manual_ingest_test.py, manual_rag_test.py, manual_retrieve_test.py, run_automation_test.py, run_basic_rag.py, run_image_test.py, run_multimodal_rag_test.py, run_router_test.py, test_ocr.py, test_text_retrieve.py, RAG_research_paper.pdf, Retrieval-Augmented-Generation-RAG.jpg, Screenshot file)
- Created PROJECT_CONTEXT.md with full v2 architecture plan
- Created PROGRESS.md (this file)
- Branch: main (v1 preserved), v2-research-rag branch to be created before coding starts

**Session 2 — Embedding Architecture**
- [x] Environment-aware embedding system implemented
- [x] BGE-M3 for local ingestion and query
- [x] Gemini fallback for production deployment
- [x] Dimension mismatch handled via separate ChromaDB per environment

**Session 3 — Environment cleanup and pipeline validation**
- Deleted venv_old
- Deleted test_dns.py, test_docling.py, app/qa/basic_rag.py
- Installed missing packages: docling, rank_bm25, streamlit
- Added ENVIRONMENT and API_URL to .env
- Tested pdf_parser, chunker, embedder end to end on attention paper
- PROGRESS.md updated to reflect actual code state
- Ready to start Phase 2: BM25 index

**Session 4 — BM25 Index Implementation**
- Created app/retrieval/bm25_index.py
- Implemented BM25Okapi wrapper with tokenization
- Added filter_type zeroing and retrieval_method tag
- Tested keyword search, filtering, and resetting
- Marked Phase 2 BM25 index as Done

**Session 5 — Hybrid Retrieval Implementation**
- Created app/retrieval/hybrid_retriever.py
- Implemented reciprocal rank fusion (RRF) for dense/sparse merge
- Implemented bge-reranker-v2-m3 using HF Inference API
- Tested RRF logic independently
- Marked Phase 2 RRF and Reranker as Done

**Session 6 — LangGraph Agent Implementation**
- Created app/agents/graph_state.py for TypedDict state
- Created app/agents/agent_graph.py with 4 tool nodes
- Implemented Tool calling at retrieval node, LangSmith tracing, reflection node with similarity threshold, logging to scholarrag.log
- Deleted obsolete router and agent files
- Marked Phase 3 as Done

**Session 7 — FastAPI Integration**
- Created app/api/main.py with all required endpoints (/ingest, /query, /health, /collection)
- Configured CORSMiddleware and applied necessary environment fixes (Torch DLL, HF symlinks)
- [x] Run `test_api.py` in both environments.
  - *Note: All 4 endpoints tested successfully (`/health`, `/query`, `/ingest`, `/collection`). Handled edge cases for Windows std-out encoding and empty PDFs effectively.*

**Session 8 — UI and Model Upgrades (2026-08-11)**
- Updated all models to Qwen 3 (qwen-qwq-32b for reasoning, qwen2.5-vl-72b-instruct for vision)
- Implemented multi-API fallback in figure_captioner.py (Groq keys 1-3, then Gemini)
- Refactored /ingest endpoint in main.py to handle multi-PDF uploads gracefully
- Wrote full Streamlit UI connecting to FastAPI endpoints with per-file status
- Reverted to LLaMA-3.3-70b (reasoning) and LLaMA-3.2-Vision — Qwen models unavailable on free Groq tier. Added think-tag stripping as safety measure.
- Fixed Docling image extraction — enabled generate_picture_images=True and PdfPipelineOptions. figures/ now populated on ingestion. Re-ingested test paper to verify figure content in ChromaDB.
- Fixed embedding_manager — removed SentenceTransformer, now using HF Inference API for bge-m3 in local environment. No local model download. Deployment-safe.
- Fixed figure_captioner — removed all Groq vision (decommissioned), switched to google-genai SDK with gemini-2.0-flash, 3 Gemini key rotation for rate limit handling

**Session 9 — LLM Upgrade to gpt-oss-120b**
- Upgraded LLM to openai/gpt-oss-120b via Groq — OpenAI open-weight MoE model, matches o4-mini on benchmarks.
- Added harmony format reasoning tag stripping in answer_node and guardrails.
- Confirmed working via direct API test.

## Current status
| Phase | Status | Notes |
|---|---|---|
| Phase 1: Local PDF & Layout Parsing | Done | Working with Docling |
| Phase 2: Structural Chunking | Done | Chunks split by semantic role (figures, tables, text) |
| Phase 3: Ingestion Pipeline | Done | Embeddings using bge-m3 via HF Inference API. ChromaDB text/metadata indexing. Figures saved locally and captioned with Gemini-2.0-flash |
| Phase 4: Hybrid Retrieval | Done | Sparse + Dense search using semantic boundaries. Integrated bge-m3 dense + BM25 sparse |
| Phase 5: Agentic Workflow | Done | LangGraph setup with specific tool calling based on query semantics |
| Phase 6: Guardrails | Done | 4 guardrails: input validation, scope check, hallucination detection, summarization detection. Dynamic chunk count based on tool calls. Integrated into FastAPI /query endpoint. |
| Phase 7: UI | Done | Fast/Streamlit frontend functional |
| Repo cleanup | Done | Test files deleted |
| PROJECT_CONTEXT.md | Done | Full architecture documented |
| Create v2 branch | Done | branch v2-research-rag exists |
| Phase 1 — Docling ingestion | Done | pdf_parser.py written and tested |
| Phase 1 — Structural chunking | Done | chunker.py written and tested |
| Phase 1 — Figure captioning | Done | figure_captioner.py written |
| Phase 1 — Metadata tagging | Done | all chunks have type, source, page, section metadata |
| Phase 2 — bge-m3 embeddings | Done | embedder.py written and tested |
| Phase 2 — ChromaDB environment-aware store | Done | Separate collections per environment, dimension validation on insert |
| Phase 2 — BM25 index | Done | Singleton pattern, filter_type zeroing, retrieval_method tag, tested |
| Phase 2 — RRF fusion | Done | Reranker via HF Inference API with graceful fallback, RRF tested |
| Phase 2 — Reranker | Done | Reranker via HF Inference API with graceful fallback, RRF tested |
| Phase 3 — LangGraph graph state | Done | Tool calling at retrieval node, LangSmith tracing, reflection node with similarity threshold, logging to scholarrag.log |
| Phase 3 — Router node | Done | Tool calling at retrieval node, LangSmith tracing, reflection node with similarity threshold, logging to scholarrag.log |
| Phase 3 — Text RAG agent | Done | Tool calling at retrieval node, LangSmith tracing, reflection node with similarity threshold, logging to scholarrag.log |
| Phase 3 — Specialized agents (Tables, Figures) | Done | Tool calling at retrieval node, LangSmith tracing, reflection node with similarity threshold, logging to scholarrag.log |
| Phase 3 — Synthesizer | Done | Tool calling at retrieval node, LangSmith tracing, reflection node with similarity threshold, logging to scholarrag.log |
| Phase 3 — Agent fallback loops | Done | Tool calling at retrieval node, LangSmith tracing, reflection node with similarity threshold, logging to scholarrag.log |
| Phase 4 — FastAPI /ingest | Done | All 4 endpoints tested, LangSmith traces confirmed, logs in scholarrag.log |
| Phase 4 — FastAPI /query | Done | All 4 endpoints tested, LangSmith traces confirmed, logs in scholarrag.log |
| Phase 4 — FastAPI /collection cleanup | Done | All 4 endpoints tested, LangSmith traces confirmed, logs in scholarrag.log |
| Phase 4 — FastAPI /health status | Done | All 4 endpoints tested, LangSmith traces confirmed, logs in scholarrag.log |
| Phase 5 — Dockerfile | Done | Preconfigured for HF Spaces |
| Phase 5 — HuggingFace Spaces deploy | Done | Ready for push |
| Phase 5 — README v2 | Done | Added deployment steps and features |

## How to use this file
At the start of every new coding session, read PROJECT_CONTEXT.md and PROGRESS.md first.
At the end of every session, update the session log and status table above.

## Decisions log
| Decision | Reason |
|---|---|
| Use case: Research Paper Assistant | Personally relevant, justifies every technical decision |
| Parser: Docling over PyMuPDF | Native table + figure extraction for scientific docs |
| Embeddings: bge-m3 over MiniLM | 8192 context, top MTEB, free via HF API |
| Retrieval: Hybrid BM25 + dense | Covers both keyword and semantic queries |
| Reranker: bge-reranker-v2-m3 | Cross-encoder precision, free, same HF API |
| Vector store: ChromaDB (keep) | Already integrated, free, persistent, no reason to switch |
| Agent framework: LangGraph | Real graph state, conditional edges, multi-step memory |
| Backend: FastAPI | Frontend-agnostic, production pattern |
| Deployment: HuggingFace Spaces | Free Docker, 16GB RAM, persistent ChromaDB volume |
| No CLIP | Queries are text-based; vision LLM descriptions work better |
| No Pinecone/Qdrant | No quality benefit at this scale, adds cost |
| No multi-tenancy now | Portfolio project, single user, add later as enhancement |
