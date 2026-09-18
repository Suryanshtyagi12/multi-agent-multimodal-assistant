import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app.agents.agent_graph import run_query

result = run_query('test query')
print('Agent graph confirmed working')
print('Reflection:', result['reflection_note'])
print('LangSmith: check smith.langchain.com for traces')
