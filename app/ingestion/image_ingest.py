# app/ingestion/image_ingest.py

import os
from PIL import Image
import pytesseract
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# Load CLIP model for image embeddings
clip_model = SentenceTransformer("clip-ViT-B-32")

# Image collection
client = chromadb.Client()
collection = client.get_or_create_collection("image_docs")

# If tesseract path ava
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract-ocr\tesseract.exe"




def ocr_image(image_path):
    """Extract text from image using OCR."""
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"OCR failed for image {image_path}: {e}")
        return ""

def ingest_image(image_path):
    """Store image embedding + OCR text into vector DB."""

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    file_name = os.path.basename(image_path)

    # 1. CLIP embedding
    embedding = clip_model.encode(img).tolist()

    # 2. OCR text
    ocr_text = ocr_image(image_path)

    # 3. Store in vector DB
    collection.add(
        documents=[ocr_text],  # OCR text stored as document
        embeddings=[embedding],  # CLIP vector
        metadatas=[{
            "source": file_name,
            "ocr_text": ocr_text,
            "path": image_path
        }],
        ids=[file_name]
    )

    print(f"Ingested image: {file_name}")
    print(f"OCR text: {ocr_text[:100]}...")
