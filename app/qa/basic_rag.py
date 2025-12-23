# app/qa/basic_rag.py

from app.retrievers.text_retriever import retrieve_text
from langchain_groq import ChatGroq



import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


def answer_query(query):
    results = retrieve_text(query)

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
You are a helpful assistant.
Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content
