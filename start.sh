#!/bin/bash
echo "=========================================="
echo "Starting ScholarRAG v2"
echo "Environment: $ENVIRONMENT"
echo "=========================================="

echo "Starting FastAPI backend on port 8000..."
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &

echo "Waiting for FastAPI to initialize..."
sleep 5

echo "Starting Streamlit frontend..."
STREAMLIT_PORT=${PORT:-7860}
streamlit run app/ui/streamlit_app.py \
    --server.port $STREAMLIT_PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false
