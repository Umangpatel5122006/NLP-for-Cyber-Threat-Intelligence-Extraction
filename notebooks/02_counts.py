import json
from collections import Counter
from pathlib import Path

import pandas as pd

path = next(Path("data/raw").rglob("SMAH-CTI.json"))
with open(path, encoding="utf-8") as f:
    df = pd.DataFrame(json.load(f))

print("Sentences:", len(df))
print("Reports:", df["doc_id"].nunique())

print("\nRelevant vs not:")
print(df["relevant"].value_counts())

labels = Counter(
    e["label"] for ents in df["entities"] if ents for e in ents
)
print("\nEntity counts:")
for label, n in labels.most_common():
    print(f"  {label}: {n}")

tactics = Counter(
    t["name"] for ts in df["tactic"] if ts for t in ts
)
print("\nTop 5 tactics:")
for name, n in tactics.most_common(5):
    print(f"  {name}: {n}")