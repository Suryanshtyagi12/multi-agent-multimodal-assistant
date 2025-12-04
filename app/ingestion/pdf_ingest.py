# app/ingestion/pdf_ingest.py

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Initialize embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create vector DB instance
client = chromadb.Client()
collection = client.get_or_create_collection("text_docs")

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start+chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def ingest_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    file_name = os.path.basename(pdf_path)

    print(f"Ingesting PDF: {file_name}")

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        for chunk in chunk_text(text):
            emb = embed_model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                metadatas=[{
                    "source": file_name,
                    "page": page_number
                }],
                ids=[f"{file_name}-{page_number}-{hash(chunk)}"],
                embeddings=[emb]
            )

    print("PDF Ingestion complete!")
