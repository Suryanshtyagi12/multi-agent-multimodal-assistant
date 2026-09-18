import requests
url = "http://localhost:8000/query"
payload = {"query": "What is the attention mechanism?"}
response = requests.post(url, json=payload)
print(response.status_code)
data = response.json()
print("Keys in response:", list(data.keys()))
print("Keys in first source:", list(data["sources"][0].keys()) if data.get("sources") else "No sources")
print("Route:", data.get("route"))
print("Reflection:", data.get("reflection_note"))
