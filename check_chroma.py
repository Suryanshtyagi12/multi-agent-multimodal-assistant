from app.retrieval.chroma_store import get_collection
col = get_collection()
results = col.get(where={'type': 'figure'}, limit=5, include=['documents', 'metadatas'])

if not results['documents']:
    print("No figure chunks found in ChromaDB.")

for i, doc in enumerate(results['documents']):
    print(f'Figure {i+1}:')
    print(f'  Content: {doc[:150]}')
    print(f'  Meta: {results["metadatas"][i]}')
