import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
response = client.chat.completions.create(
    model='openai/gpt-oss-120b',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant. Be concise.'},
        {'role': 'user', 'content': 'What is RAG in AI? One sentence.'}
    ],
    max_tokens=200,
    temperature=0
)
answer = response.choices[0].message.content

import re
for tag in ['<think>', '<reasoning>', '<thought>', '<reflection>']:
    end_tag = tag.replace('<', '</')
    pattern = f'{re.escape(tag)}.*?{re.escape(end_tag)}'
    answer = re.sub(pattern, '', answer, flags=re.DOTALL)
answer = answer.strip()

print(f'Model: openai/gpt-oss-120b')
print(f'Answer: {answer.encode("ascii", "ignore").decode("ascii")}')
print(f'Answer length: {len(answer)} chars')
print('Model working correctly' if answer else 'WARNING: empty answer')
