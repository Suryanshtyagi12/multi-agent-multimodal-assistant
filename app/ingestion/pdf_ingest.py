import os
import fitz  # PyMuPDF
from pypdf import PdfReader
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
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
    Ingest a PDF file into ChromaDB.
    OCR fallback removed for cloud compatibility (Streamlit).
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
    file_name = os.path.basename(pdf_path)
    added_chunks = 0

    # -------------------------------------------------
    # Process Pages
    # -------------------------------------------------
    for page_idx, page in enumerate(reader.pages):
        # Extract text using pypdf
        text = page.extract_text() or ""

        # Safe Fallback: If pypdf fails, skip the page (Removes Tesseract dependency)
        if not text.strip():
            print(f"⚠️ [SKIP] Page {page_idx + 1}: No extractable text found (scanned image).")
            continue

        # -------------------------------------------------
        # Chunk & Embed
        # -------------------------------------------------
        chunks = chunk_text(text)

        for chunk_idx, chunk in enumerate(chunks):
            # Ignore tiny fragments
            if len(chunk.strip()) < 30:
                continue

            # Generate vector embedding
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

    print(f"✅ [PDF INGEST] Completed")
    print(f"📦 [PDF INGEST] Total chunks added: {added_chunks}")
    print(f"💾 [PDF INGEST] Data auto-persisted by ChromaDB")