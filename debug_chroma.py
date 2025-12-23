import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collections = client.list_collections()

print("Collections found:", collections)

for col in collections:
    c = client.get_collection(col.name)
    print(f"Collection '{col.name}' count:", c.count())
