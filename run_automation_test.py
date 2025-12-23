from app.agents.automation_agent import generate_email, generate_bug_report
from app.agents.rag_agent import get_raw_context

query = "Summarize issues related to RAG systems"
context = get_raw_context(query)

print("EMAIL OUTPUT:\n")
print(generate_email(context, "Write an email to my manager summarizing key findings."))

print("\nBUG REPORT OUTPUT:\n")
print(generate_bug_report(context, "Create a bug report for issues found in the system."))
