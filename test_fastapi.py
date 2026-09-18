from fastapi.testclient import TestClient
from app.api.main import app
import sys

client = TestClient(app)

print("Test 2 - Blocked query")
response = client.post("/query", json={"query": "what is the weather today?"})
data = response.json()
print(f"Status: {response.status_code}")
print(f"Answer: {data.get('answer', '')[:80]}")
print(f"Route: {data.get('route')}")

print("\nTest 3 - Valid query")
response = client.post("/query", json={"query": "What is the attention mechanism?"})
data = response.json()
print(f"Status: {response.status_code}")
print(f"Answer: {data.get('answer', '')[:80]}")
print(f"Route: {data.get('route')}")
