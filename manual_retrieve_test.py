from app.retrievers.text_retriever import retrieve_text

res = retrieve_text("KNN", k=3)
print(res)
