# app/ingestion/image_ingest.py

import os
import re
import chromadb
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_PATH

# -------------------------------------------------
# OPTIONAL: Hard-set Tesseract path (Windows)
# -------------------------------------------------
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# -------------------------------------------------
# Models
# -------------------------------------------------
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"
IMAGE_EMBED_MODEL = "clip-ViT-B-32"

text_embedder = SentenceTransformer(TEXT_EMBED_MODEL)
image_embedder = SentenceTransformer(IMAGE_EMBED_MODEL)

# -------------------------------------------------
# OCR Text Cleaning (CRITICAL)
# -------------------------------------------------
def clean_ocr_text(text: str) -> str:
    """
    Cleans OCR noise from screenshots (Wikipedia, PDFs, UI text)
    """
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"From Wikipedia, the free encyclopedia", "", text, flags=re.I)
    return text.strip()

# -------------------------------------------------
# OCR Text Chunking
# -------------------------------------------------
def chunk_text(text, chunk_size=60, overlap=15):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# -------------------------------------------------
# Image Ingestion
# -------------------------------------------------
def ingest_image(image_path: str):
    """
    Ingest image with OCR + CLIP embeddings into ChromaDB
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\n🖼️ [IMAGE INGEST] Starting: {image_path}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    text_collection = client.get_or_create_collection("text_docs")
    image_collection = client.get_or_create_collection("image_docs")

    file_name = os.path.basename(image_path)

    # -------------------------------------------------
    # Load image
    # -------------------------------------------------
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ [IMAGE LOAD ERROR]: {e}")
        return

    # -------------------------------------------------
    # OCR Text (better config for screenshots)
    # -------------------------------------------------
    try:
        raw_text = pytesseract.image_to_string(
            img,
            config="--psm 6"
        )
        ocr_text = clean_ocr_text(raw_text)
        print(f"🔍 [OCR] Clean text length: {len(ocr_text)}")
    except Exception as e:
        print(f"❌ [OCR ERROR]: {e}")
        return

    # -------------------------------------------------
    # Store OCR text as CHUNKED embeddings
    # -------------------------------------------------
    if len(ocr_text) > 50:
        chunks = chunk_text(ocr_text)
        print(f"[DEBUG] OCR chunks created: {len(chunks)}")

        for idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue

            embedding = text_embedder.encode(chunk).tolist()

            text_collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": file_name,
                    "type": "image_ocr"
                }],
                ids=[f"{file_name}_ocr_{idx}"]
            )

        print(f"✅ [IMAGE INGEST] OCR text chunks added: {len(chunks)}")
    else:
        print("⚠️ [IMAGE INGEST] OCR text too short, skipped")

    # -------------------------------------------------
    # Store image embedding (CLIP) – optional
    # -------------------------------------------------
    try:
        image_embedding = image_embedder.encode(img).tolist()

        image_collection.add(
            documents=[file_name],
            embeddings=[image_embedding],
            metadatas=[{
                "source": file_name,
                "type": "image"
            }],
            ids=[f"{file_name}_clip"]
        )

        print("✅ [IMAGE INGEST] Image embedding stored")

    except Exception as e:
        print(f"❌ [IMAGE EMBEDDING ERROR]: {e}")

    print("💾 [IMAGE INGEST] Data auto-persisted by ChromaDB")
