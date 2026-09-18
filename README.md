

# ScholarRAG — Multi-Agent Multimodal RAG for Research Papers 📚

ScholarRAG is a production-grade, multi-agent Retrieval-Augmented Generation (RAG) system designed specifically for dense academic research papers. It handles text, tables, and complex figures natively, utilizing a dynamic LangGraph tool-routing architecture.

## 🚀 Features

- **Multimodal Ingestion**: Parses PDFs intelligently using `docling`, extracting and segmenting text, tables, and images (figures).
- **Hybrid Retrieval**: Combines sparse keyword search (BM25) and dense semantic search (`bge-m3` via ChromaDB), merged using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Uses `ms-marco-MiniLM-L-6-v2` to aggressively rerank chunks for maximum context precision.
- **Agentic Routing**: Powered by `openai/gpt-oss-120b` via Groq. A master Router node analyzes queries and conditionally calls specialized retrieval tools (`search_text`, `search_figures`, `search_tables`).
- **Vision Models**: Automatically routes extracted figures to Google Gemini 1.5 Flash for multimodal reasoning and OCR.
- **Deep Observability**: Real-time token tracking (`tokens_in`/`out`) and latency metrics displayed natively in the UI.

---

## 📊 Comprehensive Evaluation Results

ScholarRAG was rigorously evaluated against a **36-Question Golden Dataset** generated from 7 diverse research papers. The evaluation was conducted using an automated LLM Judge pipeline (`evaluation/run_evaluation.py`).

### 1. Core RAG Metrics (0.0 to 1.0 Scale)
*   **Answer Relevance (0.68):** Measures how well the generated answer directly addresses the user's query without hallucinating off-topic information.
*   **Faithfulness (0.71):** Measures how strictly the LLM adhered *only* to the retrieved context chunks (minimizing model bias/hallucination).
*   **Context Precision (0.42):** Measures whether the highest-ranked retrieved chunks contained the actual answer (Keyword overlap heuristic).
*   **Context Recall (0.63):** Measures whether the retrieval system successfully found all necessary context to fully answer the question.

### 2. Performance Breakdown by Query Difficulty
*   **EASY (Direct lookup):** 
    *   *Relevance:* 0.85 | *Faithfulness:* 0.88
*   **MEDIUM (Multi-hop reasoning within one paper):** 
    *   *Relevance:* 0.70 | *Faithfulness:* 0.75
*   **HARD (Cross-paper synthesis & complex logic):** 
    *   *Relevance:* 0.55 | *Faithfulness:* 0.60

### 3. Performance Breakdown by Modality
*   **Text Queries (9 questions):** *Relevance:* 0.75
*   **Cross-Paper Queries (9 questions):** *Relevance:* 0.58
*   **Figure Queries (4 questions):** *Relevance:* 0.78 *(Strong performance due to Google Gemini Vision OCR integration)*
*   **Table Queries (2 questions):** *Relevance:* 0.82
*   **Synthesis Queries (6 questions):** *Relevance:* 0.61

---

## 🛠️ Architecture (LangGraph)

The system utilizes a structured state-machine flow:
1. `embed_query_node`: Vectorizes the user's question using BAAI/bge-m3.
2. `tool_retrieval_node`: The LLM reads the query and selects between 1 to 4 specialized retrieval tools.
3. `reflection_node`: Analyzes the top retrieved chunk. If the ChromaDB distance score indicates poor relevance (<0.4), it forces a dynamic retry using `search_all_chunks`.
4. `answer_node`: Synthesizes the final response, injecting strict citation metadata (Filename, Page Number, Content Type).

---

## 💻 Local Setup

1. Clone the repository and configure `.env` (Requires `GROQ_API_KEY` and `GEMINI_API_KEY`).
2. Run via Docker (Recommended due to system library requirements for `docling` and `ffmpeg`):
```bash
docker build -t scholarrag .
docker run -p 8000:8000 -p 8501:8501 --env-file .env scholarrag
```
3. Access the Streamlit UI at `http://localhost:8501`.
