from app.retrieval.chroma_store import (
    get_unique_sources_count,
    get_collection_stats
)

print("TASK 1 - get_unique_sources_count")
count = get_unique_sources_count()
print(f'Unique papers in ChromaDB: {count}')
stats = get_collection_stats()
print(f'Stats unique_sources: {stats.get("unique_sources")}')
print(f'Total chunks: {stats.get("total_chunks")}')
print("\nTASK 2 - Test dynamic scaling logic manually")

def calculate_final_n(unique_papers, n_results=5):
    final_n = max(n_results, unique_papers)
    final_n = min(final_n, 15)
    fetch_n = max(20, final_n * 3)
    return final_n, fetch_n

test_cases = [1, 3, 5, 10, 15, 20]
for papers in test_cases:
    final_n, fetch_n = calculate_final_n(papers)
    print(f'Papers: {papers:2d} -> final_n: {final_n:2d} fetch_n: {fetch_n}')
