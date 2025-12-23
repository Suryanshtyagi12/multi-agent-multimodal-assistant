# app/agents/router_agent.py

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

ROUTER_PROMPT = """
You are a query classifier for a multimodal RAG system.

Classify the user's query into one of these categories:

1. TEXT_ONLY - question about text/PDFs/documents
2. IMAGE_RELATED - question about images/screenshots/diagrams
3. MULTIMODAL - question that requires BOTH text + images
4. WORKFLOW - user wants an action (email, summary, report, task list)

Respond with ONLY the label.
Query: {query}
"""

def route_query(query):
    prompt = ROUTER_PROMPT.format(query=query)
    result = llm.invoke(prompt)
    return result.content.strip()
