# app/config.py

import os

# ---------------------------
# Project root directory
# ---------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# ---------------------------
# Upload directory
# ---------------------------
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "temp_uploads")

# ---------------------------
# ChromaDB directory
# ---------------------------
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

# ---------------------------
# Ensure directories exist
# ---------------------------
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)
