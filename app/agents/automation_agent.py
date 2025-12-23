# app/agents/automation_agent.py

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_email(context, user_request):
    """
    Generates a professional email using retrieved RAG context.
    """

    prompt = f"""
You are an enterprise assistant.

Using the information below, write a professional email.

CONTEXT:
{context}

USER REQUEST:
{user_request}

Write a clear, concise, and professional email:
"""

    response = llm.invoke(prompt)
    return response.content


def generate_bug_report(context, user_request):
    """
    Generates a structured bug/incident report (Jira-style).
    """

    prompt = f"""
You are an enterprise incident management assistant.

Using the context below, generate a structured bug report in JSON format.

CONTEXT:
{context}

USER REQUEST:
{user_request}

Return ONLY valid JSON with the following fields:
- title
- description
- impact
- steps_to_reproduce
- priority
"""

    response = llm.invoke(prompt)
    return response.content


def generate_summary(context, user_request):
    """
    Generates an executive-style summary or report.
    """

    prompt = f"""
You are an AI assistant for senior management.

Using the information below, generate a concise executive summary.

CONTEXT:
{context}

USER REQUEST:
{user_request}

Summary:
"""

    response = llm.invoke(prompt)
    return response.content
