import streamlit as st
import httpx
import os
import sys
import subprocess
import time
from dotenv import load_dotenv

# --- BOOTLOADER HACK FOR HUGGINGFACE SPACES ---
# If FastAPI is not running on 8000, boot it up in the background!
try:
    httpx.get("http://127.0.0.1:8000/health", timeout=1)
except Exception:
    print("FastAPI not running. Booting it up in the background...")
    subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"])
    time.sleep(3) # Wait for it to boot

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="ScholarRAG",
    page_icon="📚",
    layout="wide"
)

with st.sidebar:
    st.title("📚 ScholarRAG")
    st.caption("Multi-Agent RAG for Research Papers")
    st.caption("Qwen3 Reasoning + Qwen Vision")
    
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5).json()
        stats = health.get("collection_stats", {})
        env = health.get("environment", "unknown")
        total = stats.get("total_chunks", 0)
        
        if total > 0:
            st.success(f"✅ {total} chunks indexed")
            col1, col2, col3 = st.columns(3)
            col1.metric("Text", stats.get("text_chunks", 0))
            col2.metric("Tables", stats.get("table_chunks", 0))
            col3.metric("Figures", stats.get("figure_chunks", 0))
            st.caption(f"Environment: {env}")
        else:
            st.info("📭 No papers indexed yet.")
    except:
        st.warning("⚠️ API not reachable")
    
    st.divider()
    st.subheader("Upload Papers")
    
    uploaded_files = st.file_uploader(
        "Select research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research papers"
    )
    
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected:")
        for f in uploaded_files:
            st.caption(f"  📄 {f.name}")
    
    upload_btn = st.button(
        "🚀 Ingest Papers",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True
    )
    
    if upload_btn and uploaded_files:
        with st.spinner(f"Processing {len(uploaded_files)} paper(s)..."):
            try:
                files_payload = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploaded_files
                ]
                response = httpx.post(
                    f"{API_URL}/ingest",
                    files=files_payload,
                    timeout=300
                )
                result = response.json()
                
                if result.get("status") == "success":
                    st.success(
                        f"✅ {result['total_files_processed']} paper(s) — "
                        f"{result['total_chunks']} chunks indexed"
                    )
                    for file_result in result.get("files", []):
                        if file_result["status"] == "success":
                            st.caption(
                                f"📄 {file_result['filename']}: "
                                f"{file_result['total_chunks']} chunks "
                                f"({file_result['text_chunks']} text, "
                                f"{file_result['table_chunks']} tables, "
                                f"{file_result['figure_chunks']} figures)"
                            )
                        elif file_result["status"] == "error":
                            st.error(
                                f"❌ {file_result['filename']}: "
                                f"{file_result.get('reason', 'error')}"
                            )
                        elif file_result["status"] == "skipped":
                            st.warning(
                                f"⚠️ {file_result['filename']} skipped: "
                                f"{file_result.get('reason', '')}"
                            )
                    st.rerun()
                else:
                    st.warning(result.get("message", "Warning during ingestion"))
            
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")
    
    st.divider()
    
    if st.button("🗑️ Clear All Papers", use_container_width=True):
        try:
            httpx.delete(f"{API_URL}/collection", timeout=10)
            st.success("Collection cleared")
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
        except Exception as e:
            st.error(f"Clear failed: {str(e)}")


st.title("Ask about your research papers")
st.caption(
    "Powered by Qwen3 reasoning + Qwen Vision | "
    "Hybrid BM25 + Dense retrieval | LangGraph agents"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)"):
                for src in msg["sources"]:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    col1.markdown(
                        f"**{src.get('source_filename', 'unknown')}**"
                    )
                    col2.caption(f"Page {src.get('page_number', '?')}")
                    col3.caption(f"`{src.get('type', '?')}`")
                    st.caption(src.get("section_title", ""))
                    st.markdown(f"> {src.get('content', '')[:200]}...")
                    st.divider()
        
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            st.caption(
                f"🔀 {meta.get('route', '?')} | "
                f"🔍 {meta.get('reflection_note', '?')} | "
                f"⏱️ {meta.get('latency_seconds', 0.0):.2f}s | "
                f"🪙 {meta.get('total_tokens', 0)} tokens"
            )

if prompt := st.chat_input(
    "Ask about methodology, results, figures, tables..."
):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Routing query and searching papers..."):
            try:
                response = httpx.post(
                    f"{API_URL}/query",
                    json={
                        "query": prompt,
                        "conversation_history": (
                            st.session_state.conversation_history
                        )
                    },
                    timeout=60
                ).json()
                
                answer = response.get("answer", "No answer returned")
                sources = response.get("sources", [])
                route = response.get("route", "")
                reflection = response.get("reflection_note", "")
                environment = response.get("environment", "")
                latency_seconds = response.get("latency_seconds", 0.0)
                token_usage = response.get("token_usage", {})
                
                st.markdown(answer)
                
                if sources:
                    with st.expander(
                        f"📎 Sources ({len(sources)} chunks)"
                    ):
                        for src in sources:
                            col1, col2, col3 = st.columns([3, 1, 1])
                            col1.markdown(
                                f"**{src.get('source_filename', 'unknown')}**"
                            )
                            col2.caption(
                                f"Page {src.get('page_number', '?')}"
                            )
                            col3.caption(f"`{src.get('type', '?')}`")
                            st.caption(src.get("section_title", ""))
                            st.markdown(
                                f"> {src.get('content', '')[:200]}..."
                            )
                            st.divider()
                
                total_tokens = token_usage.get('total_tokens', 0) if isinstance(token_usage, dict) else 0
                st.caption(
                    f"🔀 Route: {route} | "
                    f"🔍 {reflection} | "
                    f"⏱️ {latency_seconds:.2f}s | "
                    f"🪙 {total_tokens} tokens"
                )
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "meta": {
                        "route": route,
                        "reflection_note": reflection,
                        "environment": environment,
                        "latency_seconds": latency_seconds,
                        "total_tokens": total_tokens
                    }
                })
                
                st.session_state.conversation_history.append(
                    {"role": "user", "content": prompt}
                )
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": answer}
                )
                
                if len(st.session_state.conversation_history) > 10:
                    st.session_state.conversation_history = (
                        st.session_state.conversation_history[-10:]
                    )
            
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "meta": {}
                })