import urllib.request
import urllib.parse
import json

url = "https://visionquest-1.onrender.com/query"

# Test 1: Text Query
data = urllib.parse.urlencode({"text": "Who was J. Robert Oppenheimer?"}).encode("utf-8")
req = urllib.request.Request(url, data=data)

try:
    res = urllib.request.urlopen(req)
    print("=== TEST 1 (Text Query) ===")
    print("Status:", res.status)
    body_text = res.read().decode("utf-8")
    print("Body preview:", body_text[:300])
except Exception as e:
    print("=== TEST 1 ERROR ===")
    print(e)
    if hasattr(e, "read"):
        print("Error content:", e.read().decode("utf-8"))
