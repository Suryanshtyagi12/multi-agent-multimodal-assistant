import streamlit as st
import os
import sys
import chromadb
import time
import json
from PIL import Image

# -------------------------------------------------
# 1. SYSTEM PATH & CLOUD COMPATIBILITY
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

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

# Ensure directory exists for cloud storage
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_relative_path(absolute_path):
    """Fixes 'MediaFileStorageError' by converting paths for Linux servers."""
    try:
        if not os.path.isabs(absolute_path):
            return absolute_path
        rel = os.path.relpath(absolute_path, ROOT_DIR)
        return rel.replace("\\", "/") 
    except:
        return absolute_path

# -------------------------------------------------
# 2. UI STYLING (Professional Dark Mode)
# -------------------------------------------------
st.set_page_config(page_title="Multi-Agent AI Hub", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #0A0A0A !important; border-right: 1px solid #333; }
    .stMarkdown, p, h1, h2, h3, h4 { color: #FFFFFF !important; }
    div.stChatMessage { background-color: #111111; border: 1px solid #222; border-radius: 10px; }
    .stButton>button { border-radius: 5px; font-weight: bold; }
    [data-testid="stMetricValue"] { color: #58A6FF !important; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# 3. HEADER & WORKFLOW GUIDE
# -------------------------------------------------
st.title("🧠 Multi-Agent Multimodal Assistant")
st.markdown("#### *Enterprise-Grade Document Intelligence*")

cols = st.columns(3)
with cols[0]:
    st.markdown("### 1️⃣ Prepare")
    st.caption("🚀 **Ready to start?**")
    st.info("Upload your source files. Use the **Cleanup** tool in the sidebar to ensure your AI remains focused on the current project's context.")

with cols[1]:
    st.markdown("### 2️⃣ Ingest")
    st.caption("⚙️ **Building Intelligence**")
    st.success("Our agents are converting your documents into a specialized knowledge base. This creates a high-speed 'brain' for your data.")

with cols[2]:
    st.markdown("### 3️⃣ Analyze")
    st.caption("💡 **Unlock Insights**")
    st.warning("Query your data naturally. Use our **Automation Suite** to turn complex findings into professional emails and JSON reports.")

st.divider()

# -------------------------------------------------
# 4. SIDEBAR: DATA COMMAND CENTER
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Data Command")
    
    # Live Knowledge Tracker
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        colls = [c.name for c in client.list_collections()]
        count = 0
        if "text_docs" in colls: count += client.get_collection("text_docs").count()
        if "image_docs" in colls: count += client.get_collection("image_docs").count()
        st.metric("Stored Knowledge Chunks", count)
    except:
        st.metric("Knowledge Chunks", "Syncing...")

    st.subheader("📤 Upload Knowledge")
    up_pdfs = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True)
    up_imgs = st.file_uploader("Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if st.button("📥 Start Ingestion", use_container_width=True, type="primary"):
        if not up_pdfs and not up_imgs:
            st.warning("Please upload files first.")
        else:
            with st.status("🏗️ Agent Ingestion in Progress...", expanded=True) as status:
                if up_pdfs:
                    for pdf in up_pdfs:
                        path = os.path.join(UPLOAD_DIR, pdf.name)
                        with open(path, "wb") as f: f.write(pdf.getbuffer())
                        ingest_pdf(path)
                        st.write(f"✅ Indexed Text: {pdf.name}")
                if up_imgs:
                    for img in up_imgs:
                        path = os.path.join(UPLOAD_DIR, img.name)
                        with open(path, "wb") as f: f.write(img.getbuffer())
                        ingest_image(path)
                        st.write(f"✅ Indexed Visuals: {img.name}")
                status.update(label="Ingestion Complete!", state="complete")
            st.rerun()

    st.divider()
    st.markdown("### 🧹 Database Cleanup")
    st.caption("Clean the data after use to maintain agent accuracy.")
    if st.button("Clear Vector Database", use_container_width=True):
        for c_name in ["text_docs", "image_docs", "multimodal_collection"]:
            try: client.delete_collection(c_name)
            except: pass
        st.session_state.last_rag_response = None
        st.session_state.last_query = ""
        st.success("Brain reset complete.")
        time.sleep(1)
        st.rerun()

# -------------------------------------------------
# 5. MAIN CHAT & RESULTS
# -------------------------------------------------
if "last_rag_response" not in st.session_state:
    st.session_state.last_rag_response = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

query = st.chat_input("Ask about your documents...")

if query:
    with st.spinner("🤖 Consulting Specialist Agents..."):
        st.session_state.last_route = route_query(query)
        st.session_state.last_rag_response = multimodal_rag(query)
        st.session_state.last_query = query

if st.session_state.last_rag_response:
    res = st.session_state.last_rag_response
    
    # Display Router Badge
    route = st.session_state.get('last_route', 'GENERAL')
    st.markdown(f"**Agent Routed to:** `:blue[{route.upper()}]`")
    
    with st.chat_message("assistant"):
        st.markdown(res.get("answer", "No answer found."))

    # --- COLLAPSIBLE EVIDENCE SECTION ---
    st.markdown("### 🔍 Source Verification")
    
    with st.expander("📄 View Text Evidence", expanded=False):
        text_chunks = res.get("text", [])
        if text_chunks:
            for txt in text_chunks: 
                st.caption(f"📌 {txt}")
                st.markdown("---")
        else:
            st.write("No direct text evidence found.")
    
    with st.expander("🖼 View Image Evidence", expanded=False):
        image_paths = res.get("images", [])
        if image_paths:
            img_cols = st.columns(len(image_paths))
            for i, p in enumerate(image_paths):
                rel_p = get_relative_path(p)
                try:
                    img_file = Image.open(rel_p)
                    img_cols[i].image(img_file, use_container_width=True, caption=f"Source: {os.path.basename(p)}")
                except:
                    img_cols[i].error(f"Missing File: {os.path.basename(p)}")
        else:
            st.write("No relevant visual evidence found.")

    # -------------------------------------------------
    # 6. AUTOMATION CENTER
    # -------------------------------------------------
    st.markdown("---")
    st.markdown("### 🛠️ Automation Center")
    raw_ctx = get_raw_context(st.session_state.last_query)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("✉️ Draft Email", use_container_width=True):
            st.markdown("#### Drafted Email")
            st.info(generate_email(raw_ctx, "Draft a professional email."))
            
    with c2:
        if st.button("📝 Executive Summary", use_container_width=True):
            st.markdown("#### Summary")
            st.success(generate_summary(raw_ctx, "Provide a concise summary."))
            
    with c3:
        if st.button("🐞 Bug Report", use_container_width=True):
            st.markdown("#### Technical JSON Report")
            bug_output = generate_bug_report(raw_ctx, "Format as valid JSON.")
            
            clean_json = bug_output.replace("```json", "").replace("```", "").strip()
            try:
                bug_data = json.loads(clean_json)
                st.json(bug_data)
            except:
                st.warning("AI output was not perfectly formatted JSON. Raw text below:")
                st.code(bug_output)

# -------------------------------------------------
# Debugging
# -------------------------------------------------
with st.expander("🛠 System Debug"):
    st.write("Root Directory:", ROOT_DIR)
    st.write("Chroma Path:", CHROMA_PATH)
