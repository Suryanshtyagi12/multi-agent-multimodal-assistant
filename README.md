
# Multi-Agent Multimodal Enterprise RAG Assistant

Demo:- https://suryanshtyagi12-multi-agent-multimoda-appuistreamlit-app-kg5o6o.streamlit.app/

A production-style **Multi-Agent, Multimodal Retrieval-Augmented Generation (RAG) system**
that can ingest **PDFs and images**, understand them using **text + vision models**, and
generate **grounded answers and enterprise outputs** such as summaries, emails, and bug reports.

This project demonstrates real-world **GenAI system design**, including persistent vector
databases, multimodal embeddings, OCR, agent orchestration, and an interactive chat UI.

---

## 🚀 Key Features

- **Multimodal RAG**: Query across PDFs, documents, screenshots, and diagrams  
- **Text + Image Retrieval**:
  - Text embeddings for PDFs
  - CLIP-based image embeddings for visual search
  - OCR (Tesseract) for extracting text from images
- **Multi-Agent Architecture**:
  - Router Agent (query classification)
  - RAG Agent (multimodal retrieval + reasoning)
  - Automation Agent (email, summary, bug report generation)
- **Persistent Vector Database**:
  - ChromaDB with disk persistence
- **Free LLM Integration**:
  - Groq LLaMA-3.1 (no paid API required)
- **Interactive Chat UI**:
  - Built with Streamlit
  - Upload files, ask questions, view evidence, trigger automations

---

## 🧠 System Architecture

User Query
↓
Streamlit UI
↓
Router Agent
↓
Multimodal RAG Agent
├── Text Retriever (PDFs)
├── Image Retriever (CLIP + OCR)
↓
LLM (Groq LLaMA-3.1)
↓
Answer / Summary / Email / Bug Report


---

## 📂 Project Structure

multi-agent-multimodal-assistant/
│
├── app/
│ ├── ingestion/
│ │ ├── pdf_ingest.py
│ │ ├── image_ingest.py
│ ├── retrievers/
│ │ ├── text_retriever.py
│ │ ├── image_retriever.py
│ ├── agents/
│ │ ├── router_agent.py
│ │ ├── rag_agent.py
│ │ ├── automation_agent.py
│ ├── ui/
│ │ ├── streamlit_app.py
│
├── chroma_db/ # Persistent vector store
├── run_basic_rag.py
├── run_image_test.py
├── run_multimodal_rag_test.py
├── test_text_retrieve.py
├── requirements.txt
├── README.md


---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
2️⃣ Activate Virtual Environment
Windows (PowerShell):

venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Set Environment Variables
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here
Do NOT commit .env to GitHub.

▶️ Running the Project
Ingest PDFs
python run_basic_rag.py
Ingest Images (OCR + CLIP)
python run_image_test.py
Test Text Retrieval
python test_text_retrieve.py
Run Multimodal RAG (CLI Test)
python run_multimodal_rag_test.py
💬 Run the Chatbot UI (Recommended)
streamlit run app/ui/streamlit_app.py
Then open the browser URL shown by Streamlit.

In the UI you can:
Upload PDFs and images

Ask questions in chat

See text + image evidence

Generate:

Email drafts

Executive summaries

Bug reports (JSON)



