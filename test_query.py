import urllib.request
import json

url = 'http://localhost:8000/query'
data = json.dumps({'query': 'what is attention is all you need paper about?'}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        print('--- FULL JSON RESPONSE ---')
        print(json.dumps(res, indent=2))
        ans = res.get('answer', '')
        print('\n--- DIAGNOSTIC RESULTS ---')
        print(f'1. Answer field has text: {bool(ans.strip())}')
        print(f'2. Answer text length: {len(ans)}')
        print(f'3. Contains errors: {"error" in res}')
except Exception as e:
    print('HTTP Request Error:', e)
