---
# Project context — Research Paper RAG Assistant v2

## What this project is
A multimodal RAG system specifically designed for research papers. It can parse text, tables, and figures from academic PDFs, understand figure content using a vision LLM, and answer questions using a true multi-agent LangGraph system. Fully free to run and deploy.

## Use case
Research Paper Assistant — users upload academic PDFs, ask questions about methodology, results tables, figures, and citations. The system routes each query to the appropriate specialist agent.

## Why this use case
Research papers have inherently different content types (text, tables, figures, equations) that require different retrieval strategies. This justifies the multi-agent architecture in a way that is technically defensible in interviews.

## Branch strategy
- `main` — original v1 code (preserved for interview narrative)
- `v2-research-rag` — all upgrades happen here

## Current state (v1 — what exists)
- PDF ingestion: PyMuPDF text-only, no table or figure support
- Chunking: fixed-size character splitting
- Embeddings: all-MiniLM-L6-v2 (256 token context, low MTEB)
- Retrieval: dense cosine similarity top-k only
- Agents: not real agents — functions calling Groq with different prompts
- Router: one LLM call returning a string label (not a graph)
- UI: Streamlit only, no API layer
- Deployment: local only

## Target state (v2 — what we are building)
### Ingestion
- Parser: Docling (IBM open source) — handles text, tables, figures natively
- Tables: converted to markdown strings, stored with type="table" metadata
- Figures: extracted image sent to Groq LLaMA-3.2-Vision → text description stored with type="figure" metadata
- Chunking: structural (by section boundary) + sentence-window for dense paragraphs. Tables and figure+caption never split.
- Metadata per chunk: type, source_filename, page_number, section_title

### Embeddings + retrieval
- Embedding model: BAAI/bge-m3 via HuggingFace Inference API (free, 8192 token context, top MTEB)
- Sparse index: rank_bm25 alongside ChromaDB
- Fusion: reciprocal rank fusion (RRF) to merge dense + sparse results
- Reranker: bge-reranker-v2-m3 via HuggingFace Inference API — rescores top-20 → final top-5
- Vector store: ChromaDB (keep, no change)

### Agent system (LangGraph)
Real LangGraph graph with 5 nodes:
1. Router node — classifies query type, sets conditional edge
2. Text RAG agent — queries ChromaDB with where={"type": "text"}
3. Table agent — queries ChromaDB with where={"type": "table"}
4. Figure agent — queries ChromaDB with where={"type": "figure"}
5. Synthesis agent — called for multi-part queries, merges outputs from multiple agents

Graph state carries: query, route, retrieved_chunks, agent_outputs, conversation_history

### Backend + UI
- FastAPI backend with endpoints:
  - POST /ingest — accepts PDF, runs full pipeline, returns summary
  - POST /query — accepts text query, runs agent graph, returns answer + source chunks
  - DELETE /collection — wipes ChromaDB + BM25 index (cleanup button)
  - GET /health — health check for deployment
- Streamlit UI refactored to call FastAPI endpoints only (no direct logic)

### Deployment
- Platform: HuggingFace Spaces (free, 16GB RAM Docker)
- LLM: Groq openai/gpt-oss-120b (free tier)
- Vision: Groq LLaMA-3.2-Vision (free tier)
- Embeddings: HuggingFace Inference API bge-m3 (free tier)
- Reranker: HuggingFace Inference API bge-reranker-v2-m3 (free tier)
- Total cost: $0/month

## Target folder structure (v2)
research-paper-rag/
├── app/
│   ├── ingestion/
│   │   ├── pdf_parser.py        # Docling parsing → dict export
│   │   ├── chunker.py           # Structural + sentence-window chunking
│   │   ├── figure_captioner.py  # Groq Vision → text description
│   │   └── embedder.py          # bge-m3 via HF Inference API
│   ├── retrieval/
│   │   ├── chroma_store.py      # ChromaDB operations
│   │   ├── bm25_index.py        # rank_bm25 sparse index
│   │   ├── hybrid_retriever.py  # RRF fusion + reranker
│   ├── agents/
│   │   ├── graph_state.py       # LangGraph state definition
│   │   ├── router_node.py       # Real routing node
│   │   ├── text_rag_agent.py    # Text retrieval agent
│   │   ├── table_agent.py       # Table retrieval agent
│   │   ├── figure_agent.py      # Figure retrieval agent
│   │   ├── synthesis_agent.py   # Multi-output synthesis
│   │   └── agent_graph.py       # Full LangGraph graph assembly
│   ├── api/
│   │   └── main.py              # FastAPI app with all endpoints
│   └── ui/
│       └── streamlit_app.py     # UI calls FastAPI only
├── Dockerfile                   # Runs FastAPI + Streamlit together
├── requirements.txt
├── .env.example
├── PROJECT_CONTEXT.md           # This file
├── PROGRESS.md                  # Updated after every session
└── README.md

## Tech stack
| Component | Technology | Rationale |
|---|---|---|
| PDF Parser | Docling | Preserves tables natively, extracts embedded images |
| Embeddings | BGE-M3 | Strong multimodal semantics, 8192 context window |
| Vector Store | ChromaDB | Local persistent vector storage |
| Sparse Index | BM25 (rank_bm25) | Exact keyword matching for terminology |
| Reranker | BAAI/bge-reranker-v2-m3 | High-precision cross-encoder via HF Inference API |
| LLM | Groq openai/gpt-oss-120b | OpenAI open-weight MoE, matches o4-mini on benchmarks, free via Groq |
| Vision LLM | Groq LLaMA-3.2-Vision | Free tier |
| Orchestration| LangGraph | Reliable state machine for multi-agent loops |
| API Layer | FastAPI | Extensible, standard Python backend |
| Frontend | Streamlit | Fast prototyping, simple file upload UI |
| Deployment | HuggingFace Spaces | Free Docker, 16GB RAM, persistent volume |

## Embedding Strategy
- Local: BAAI/bge-m3 via sentence-transformers (1024 dims)
- Production: Gemini text-embedding-004 (768 dims)
- Environment controlled via ENVIRONMENT variable
- ChromaDB must be rebuilt per environment — dims differ

## Key interview talking points
1. Why Docling? Built by IBM Research for scientific documents specifically. Preserves table structure, extracts figures with captions, maintains reading order in multi-column layouts.
2. Why bge-m3 over MiniLM? MiniLM has 256 token context — truncates most research paper chunks. bge-m3 has 8192 context and ranks in MTEB top 3 for scientific text.
3. Why hybrid retrieval? Pure vector search fails on exact queries like paper IDs or author names. BM25 handles these. RRF merges both without needing a trained fusion model.
4. Why LangGraph over if-else? Multi-step queries require state flow between agents. LangGraph handles conditional edges, fallback routing, and conversation memory. if-else cannot.
5. Why FastAPI alongside Streamlit? The RAG backend should be frontend-agnostic. Any frontend (React, mobile) can use the same /query endpoint.
6. Why this use case? I built it to solve my own problem — understanding dense ML papers with tables and figures. I know the pain point personally.
---
