from app.guardrails.guardrails import run_input_guardrails

tests = [
    ('', True),
    ('What is the methodology?', True),
    ('What is the weather today?', True),
    ('summarize the paper', True),
    ('What is the attention mechanism?', False),
]

for query, has_docs in tests:
    result = run_input_guardrails(query, has_docs)
    print(f'Query: "{query[:40]}"')
    print(f'  allowed: {result["allowed"]}')
    print(f'  message: {result["message"][:80]}')
    print(f'  is_summarization: {result["is_summarization"]}')
    print()
