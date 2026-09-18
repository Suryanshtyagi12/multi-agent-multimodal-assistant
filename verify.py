import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("--- TASK 3 ---")
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

print("\n--- TASK 4 ---")
from docling.document_converter import DocumentConverter
print('Docling OK')

print("\n--- TASK 5 ---")
from app.ingestion.pdf_parser import parse_pdf
elements = parse_pdf('test_paper.pdf')
print(f'Total elements: {len(elements)}')
types = {}
for e in elements:
    types[e['type']] = types.get(e['type'], 0) + 1
print(f'Breakdown: {types}')

print("\n--- TASK 6 ---")
from app.ingestion.embedder import embed_text
vec = embed_text('What is the attention mechanism?')
print(f'Embedding dim: {len(vec)}')
print('Embedder OK')
