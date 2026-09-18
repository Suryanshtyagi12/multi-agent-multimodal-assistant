import json
try:
    with open('evaluation_dataset.json', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Total questions: {len(data["questions"])}')
    print(f'Keys in each question: {list(data["questions"][0].keys())}')
    print()
    for q in data['questions'][:3]:
        print(f'ID: {q["id"]}')
        print(f'Question: {q["question"][:80]}')
        print(f'Ground truth: {q["ground_truth"][:80]}')
        print(f'Context type: {q["context_type"]}')
        print(f'Test type: {q["test_type"]}')
        print()
except Exception as e:
    print(f"Error reading dataset: {e}")

from app.retrieval.chroma_store import get_collection_stats, get_unique_sources_count
stats = get_collection_stats()
sources = get_unique_sources_count()
print("CHROMA STATS:")
print(f'Total chunks: {stats["total_chunks"]}')
print(f'Unique papers: {sources}')
print(f'Text chunks: {stats["text_chunks"]}')
print(f'Table chunks: {stats["table_chunks"]}')
print(f'Figure chunks: {stats["figure_chunks"]}')
