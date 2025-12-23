# app/agents/rag_agent.py

from app.retrievers.text_retriever import retrieve_text
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------------------------
# Multimodal RAG (TEXT + IMAGE OCR via TEXT)
# -------------------------------------------------
def multimodal_rag(query):
    """
    Unified RAG using:
    - PDF text
    - Image OCR text (stored in text_docs)
    """

    print("\n[DEBUG] Running multimodal RAG...")

    # ---------------------
    # 1. Retrieve TEXT (PDF + Image OCR)
    # ---------------------
    results = retrieve_text(query, k=8)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return {
            "answer": "No relevant information found in uploaded documents or images.",
            "text": [],
            "images": []
        }

    context_blocks = []
    image_evidence = []

    for doc, meta in zip(documents, metadatas):
        source_type = meta.get("type", "pdf")
        source_name = meta.get("source", "unknown")

        if source_type == "image_ocr":
            context_blocks.append(
                f"[IMAGE OCR TEXT | {source_name}]\n{doc}"
            )
            image_evidence.append(source_name)
        else:
            context_blocks.append(
                f"[PDF TEXT | {source_name}]\n{doc}"
            )

    context = "\n\n".join(context_blocks)

    # ---------------------
    # 2. Build prompt
    # ---------------------
    prompt = f"""
You are a multimodal RAG assistant.

Answer the question ONLY using the evidence below.
If the answer is not present, say "Not found in the provided documents."

====================
EVIDENCE:
{context}
====================

QUESTION:
{query}

ANSWER:
"""

    print("[DEBUG] Sending context to LLM...")

    response = llm.invoke(prompt)

    print("[DEBUG] Answer generated.")

    return {
        "answer": response.content,
        "text": documents,
        "images": list(set(image_evidence))
    }


# -------------------------------------------------
# Raw Context (Automation tools)
# -------------------------------------------------
def get_raw_context(query):
    results = retrieve_text(query, k=8)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    blocks = []

    for doc, meta in zip(documents, metadatas):
        source_type = meta.get("type", "pdf")
        source = meta.get("source", "unknown")
        blocks.append(f"[{source_type.upper()} | {source}]\n{doc}")

    return "\n\n".join(blocks)
