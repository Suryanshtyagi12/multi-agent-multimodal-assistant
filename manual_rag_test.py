from app.agents.rag_agent import multimodal_rag

res = multimodal_rag("Explain the KNN model used in the document")
print(res["answer"])
