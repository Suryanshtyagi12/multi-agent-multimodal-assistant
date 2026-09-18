import urllib.request
import os
import traceback

print("--- TASK 7: Ingestion pipeline end to end ---")
try:
    if not os.path.exists('test_paper.pdf'):
        urllib.request.urlretrieve('https://arxiv.org/pdf/1706.03762', 'test_paper.pdf')
        print('Downloaded attention is all you need paper')
    from app.ingestion.pdf_parser import parse_pdf
    elements = parse_pdf('test_paper.pdf')
    print(f'Total elements parsed: {len(elements)}')
    types = {}
    for e in elements:
        types[e['type']] = types.get(e['type'], 0) + 1
    print(f'Element breakdown: {types}')
    print('First text element:')
    print(elements[0]['content'][:200])
except Exception as e:
    print("Error in Task 7:")
    traceback.print_exc()

print("\n--- TASK 8: Test chunker ---")
try:
    from app.ingestion.chunker import chunk_pdf
    chunks = chunk_pdf('test_paper.pdf')
    print(f'Total chunks: {len(chunks)}')
    methods = {}
    chunk_types = {}
    for c in chunks:
        methods[c.get('chunking_method', 'unknown')] = methods.get(c.get('chunking_method', 'unknown'), 0) + 1
        chunk_types[c['type']] = chunk_types.get(c['type'], 0) + 1
    print(f'Chunk types: {chunk_types}')
    print(f'Chunking methods used: {methods}')
except Exception as e:
    print("Error in Task 8:")
    traceback.print_exc()

print("\n--- TASK 9: Test embedder ---")
try:
    from app.ingestion.embedder import embed_text
    vec = embed_text('What is the attention mechanism?')
    print(f'Embedding dimension: {len(vec)}')
    print(f'First 3 values: {vec[:3]}')
    print('Embedder working correctly')
except Exception as e:
    print("Error in Task 9:")
    traceback.print_exc()
