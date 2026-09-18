---
title: Scholarrag
emoji: 🦀
colorFrom: green
colorTo: green
sdk: gradio
pinned: false
---

# ScholarRAG — Multi-Agent RAG for Research Papers

A multimodal RAG system for academic research papers. Upload PDFs,
ask questions about text, tables, and figures using a true
multi-agent LangGraph system with openai/gpt-oss-120b reasoning.

## Live Demo
[Available on HuggingFace Spaces]

## What is new in v2 vs v1

| Component | v1 | v2 |
|---|---|---|
| PDF parser | PyMuPDF text only | Docling — text + tables + figures |
| Chunking | Fixed 512 chars | Structural + semantic (800 word threshold) |
| Embeddings | MiniLM 256 tokens | bge-m3 8192 tokens via HF Inference API |
| Retrieval | Dense only | Hybrid BM25 + dense + cross-encoder reranker |
| Agents | Prompt functions | Real LangGraph tool-calling agent graph |
| LLM | Various | openai/gpt-oss-120b via Groq |
| Vision | None | LLaMA Vision with 3 Groq key + Gemini fallback |
| Backend | None | FastAPI with /ingest /query /health endpoints |
| Multi-PDF | Single file | Multiple PDFs in one request |
| Deployment | Local only | HuggingFace Spaces Docker — free |

## Architecture

