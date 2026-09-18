import os
import subprocess
import time
import sys

print("Starting ScholarRAG via Gradio SDK Trick...")

# 1. Start the FastAPI backend
fastapi_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
)

# Wait for backend to spin up
time.sleep(5)

# 2. Start the Streamlit frontend on port 7860 (the port HuggingFace expects)
streamlit_process = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py",
        "--server.port", "7860",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false"
    ]
)

# Keep the main thread alive
fastapi_process.wait()
streamlit_process.wait()
