
"""
It is for explaoring dataset instance in json formate
"""


import json
from pathlib import Path

path = next(Path("data/raw").rglob("SMAH-CTI.json"))
print("Found:", path)

with open(path, encoding="utf-8") as f:
    data = json.load(f)

print("Type:", type(data).__name__)
print("Size:", len(data))

first = data[0] if isinstance(data, list) else next(iter(data.values()))
print(json.dumps(first, indent=2, ensure_ascii=False)[:1500])


print("\n--- A relevant sentence ---")
rel = next(r for r in data if r["relevant"] == "yes" and r["entities"])
print(json.dumps(rel, indent=2, ensure_ascii=False)[:2000])