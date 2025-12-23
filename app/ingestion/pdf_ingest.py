# app/ingestion/pdf_ingest.py

import os
import fitz  # PyMuPDF
from pypdf import PdfReader
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer

# OCR (safe import)
import pytesseract

# -------------------------------------------------
# OPTIONAL: Hard-set Tesseract path (Windows safety)
# -------------------------------------------------
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

from app.config import CHROMA_PATH

# -------------------------------------------------
# Embedding Model (MUST MATCH RETRIEVER)
# -------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# -------------------------------------------------
# Text Chunking
# -------------------------------------------------
def chunk_text(text, chunk_size=400, overlap=80):
    """
    Split text into overlapping chunks
    """
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
# PDF Ingestion
# -------------------------------------------------
def ingest_pdf(pdf_path: str):
    """
    Ingest a PDF file into ChromaDB with OCR fallback.
    Compatible with ChromaDB >= 0.4.x (auto-persist).
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n📄 [PDF INGEST] Starting ingestion: {pdf_path}")

    # -------------------------------------------------
    # ChromaDB Client (AUTO-PERSIST)
    # -------------------------------------------------
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="text_docs"
    )

    print(f"📦 [CHROMA] Using path: {CHROMA_PATH}")
    print(f"📦 [CHROMA] Existing docs: {collection.count()}")

    # -------------------------------------------------
    # PDF Readers
    # -------------------------------------------------
    reader = PdfReader(pdf_path)
    ocr_doc = fitz.open(pdf_path)

    file_name = os.path.basename(pdf_path)
    added_chunks = 0

    # -------------------------------------------------
    # Process Pages
    # -------------------------------------------------
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        # OCR fallback
        if not text.strip():
            try:
                print(f"🔍 [OCR] Page {page_idx + 1}")
                pix = ocr_doc.load_page(page_idx).get_pixmap()
                img = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples
                )
                text = pytesseract.image_to_string(img)
            except Exception as e:
                print(f"❌ [OCR ERROR] Page {page_idx + 1}: {e}")
                continue

        if not text.strip():
            continue

        # -------------------------------------------------
        # Chunk & Embed
        # -------------------------------------------------
        chunks = chunk_text(text)

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue

            embedding = embed_model.encode(chunk).tolist()

            doc_id = f"{file_name}_p{page_idx}_c{chunk_idx}"

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": file_name,
                    "page": page_idx + 1,
                    "type": "pdf"
                }],
                ids=[doc_id]
            )

            added_chunks += 1

    # -------------------------------------------------
    # NO persist() call (ChromaDB auto-persists)
    # -------------------------------------------------
    print(f"✅ [PDF INGEST] Completed")
    print(f"📦 [PDF INGEST] Total chunks added: {added_chunks}")
    print(f"💾 [PDF INGEST] Data auto-persisted by ChromaDB")
