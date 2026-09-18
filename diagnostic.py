import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

print('Testing Groq LLaMA-3.3-70b directly...')
try:
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant. Answer concisely.'
            },
            {
                'role': 'user',
                'content': 'What is the attention mechanism in transformers? Answer in 3 sentences.'
            }
        ],
        max_tokens=500
    )
    answer = response.choices[0].message.content
    print(f'Answer length: {len(answer)}')
    print(f'Answer: {answer[:300]}')
except Exception as e:
    print(f'Groq call failed: {type(e).__name__}: {e}')

print()
print('Testing agent_graph run_query directly...')
try:
    from app.agents.agent_graph import run_query
    result = run_query('What is attention mechanism?')
    print(f'Answer length: {len(result["answer"])}')
    print(f'Answer: {result["answer"][:300]}')
    print(f'Route: {result["route"]}')
    print(f'Reflection: {result["reflection_note"]}')
    print(f'Sources: {len(result["sources"])}')
except Exception as e:
    print(f'run_query failed: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
