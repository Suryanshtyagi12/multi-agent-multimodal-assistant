# run_basic_rag.py

from app.ingestion.pdf_ingest import ingest_pdf
from app.qa.basic_rag import answer_query

# Step 1: Ingest a sample PDF
ingest_pdf("RAG_research_paper.pdf")   # make sure sample.pdf is in the root folder

# Step 2: Ask a question
print(answer_query("What is this document about?"))


