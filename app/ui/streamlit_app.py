# app/ui/streamlit_app.py

import streamlit as st
import os
import sys
import chromadb  # Added for safe collection-level clearing

# -------------------------------------------------
# Fix Python Path (IMPORTANT)
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# -------------------------------------------------
# Imports
# -------------------------------------------------
from app.config import UPLOAD_DIR, CHROMA_PATH
from app.ingestion.pdf_ingest import ingest_pdf
from app.ingestion.image_ingest import ingest_image

from app.agents.router_agent import route_query
from app.agents.rag_agent import multimodal_rag, get_raw_context
from app.agents.automation_agent import (
    generate_email,
    generate_summary,
    generate_bug_report
)

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Multimodal RAG Assistant",
    layout="wide"
)

st.title("🧠 Multi-Agent Multimodal RAG Assistant")
st.info("📌 Upload PDFs or Images → Click **Ingest Uploaded Files** → Ask questions")

# -------------------------------------------------
# Session State (Persistent Tracking)
# -------------------------------------------------
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

if "last_rag_response" not in st.session_state:
    st.session_state.last_rag_response = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# -------------------------------------------------
# Sidebar: Upload Knowledge & DB Management
# -------------------------------------------------
st.sidebar.header("📂 Knowledge Management")

st.sidebar.caption(f"📁 Upload dir: `{UPLOAD_DIR}`")
st.sidebar.caption(f"🧠 Chroma dir: `{CHROMA_PATH}`")

uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

uploaded_images = st.sidebar.file_uploader(
    "Upload Image files",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# --- Ingest Button ---
if st.sidebar.button("📥 Ingest Uploaded Files"):
    if not uploaded_pdfs and not uploaded_images:
        st.sidebar.error("❌ No files uploaded.")
    else:
        st.sidebar.info("🔄 Processing files...")
        
        # Filter for only new files
        new_pdfs = [f for f in (uploaded_pdfs or []) if f.name not in st.session_state.ingested_files]
        new_imgs = [f for f in (uploaded_images or []) if f.name not in st.session_state.ingested_files]
        
        total_new = len(new_pdfs) + len(new_imgs)
        
        if total_new == 0:
            st.sidebar.warning("⚠️ All selected files are already in the database.")
        else:
            progress = st.sidebar.progress(0)
            completed = 0

            # Ingest PDFs
            for pdf in new_pdfs:
                pdf_path = os.path.join(UPLOAD_DIR, pdf.name)
                with open(pdf_path, "wb") as f:
                    f.write(pdf.getbuffer())
                ingest_pdf(pdf_path)
                st.session_state.ingested_files.add(pdf.name)
                completed += 1
                progress.progress(completed / total_new)

            # Ingest Images
            for img in new_imgs:
                img_path = os.path.join(UPLOAD_DIR, img.name)
                with open(img_path, "wb") as f:
                    f.write(img.getbuffer())
                ingest_image(img_path)
                st.session_state.ingested_files.add(img.name)
                completed += 1
                progress.progress(completed / total_new)

            st.sidebar.success(f"✅ Successfully added {completed} new files!")

# --- Safe Clear DB Button (Windows Compatible) ---
st.sidebar.divider()
st.sidebar.subheader("🗑 Database Control")
if st.sidebar.button("Clear Vector Database"):
    try:
        # Connect to client to delete collections internally
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Attempt to delete common collection names
        # Add any other specific collection names you use in your ingestion scripts
        for coll_name in ["text_docs", "image_docs", "multimodal_collection"]:
            try:
                client.delete_collection(coll_name)
            except Exception:
                pass # Collection doesn't exist
        
        # Reset local tracking
        st.session_state.ingested_files.clear()
        st.session_state.last_rag_response = None
        st.session_state.last_query = ""
        
        st.sidebar.success("✅ Database cleared safely (collections deleted).")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to clear database: {e}")

# -------------------------------------------------
# Main Chat Interface
# -------------------------------------------------
st.subheader("💬 Ask a Question")

user_query = st.text_input(
    "Enter your question:",
    placeholder="Ask something from the uploaded documents...",
    value=st.session_state.last_query
)

if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        st.session_state.last_query = user_query
        with st.spinner("🤔 Analyzing data..."):
            # Route and RAG
            st.session_state.last_route = route_query(user_query)
            st.session_state.last_rag_response = multimodal_rag(user_query)

# -------------------------------------------------
# Display Results
# -------------------------------------------------
if st.session_state.last_rag_response:
    res = st.session_state.last_rag_response
    
    st.info(f"🔀 Router Decision: **{st.session_state.get('last_route', 'N/A')}**")
    
    st.markdown("### ✅ Answer")
    st.write(res.get("answer", "No answer generated."))

    # Evidence
    col_text, col_img = st.columns(2)
    with col_text:
        with st.expander("📄 Text Evidence", expanded=True):
            text_chunks = res.get("text", [])
            if text_chunks:
                for chunk in text_chunks:
                    st.markdown(f"- {chunk}")
            else:
                st.write("No text evidence found.")

    with col_img:
        with st.expander("🖼 Image Evidence", expanded=True):
            images = res.get("images", [])
            if images:
                for img in images:
                    try:
                        st.image(img, use_container_width=True)
                    except:
                        st.write(f"Source: {img}")
            else:
                st.write("No image evidence found.")

    # ---------------------------
    # Automation Tools
    # ---------------------------
    st.markdown("---")
    st.markdown("### ⚙️ Automation Tools")
    
    raw_context = get_raw_context(st.session_state.last_query)
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("✉️ Generate Email"):
            email = generate_email(raw_context, "Write a professional email.")
            st.text_area("Email Draft", email, height=250)

    with c2:
        if st.button("📝 Generate Summary"):
            summary = generate_summary(raw_context, "Create an executive summary.")
            st.text_area("Summary", summary, height=250)

    with c3:
        if st.button("🐞 Generate Bug Report"):
            bug = generate_bug_report(raw_context, "Create a bug report in JSON format.")
            st.text_area("Bug Report", bug, height=250)

# -------------------------------------------------
# Footer Debug Info
# -------------------------------------------------
with st.expander("🛠 Debug Info"):
    st.write("UPLOAD_DIR:", UPLOAD_DIR)
    st.write("CHROMA_PATH:", CHROMA_PATH)
    st.write("Ingested files count:", len(st.session_state.ingested_files))
    st.write("File list:", list(st.session_state.ingested_files))