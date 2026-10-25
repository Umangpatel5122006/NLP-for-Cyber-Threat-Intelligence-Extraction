import json
from collections import defaultdict

from seqeval.metrics import classification_report, f1_score

from cti.core.dataset import load_smah
from cti.models.crf import CRFTagger
from cti.models.data import load_bio

LAYER = "entities"
tx, ty = load_bio(LAYER, "train")
model = CRFTagger()
model.fit(tx, ty)

dev = load_smah()["dev"]
source_of = dict(zip(dev["uid"], dev["source"]))
with open(f"data/processed/bio_en_{LAYER}_dev.json", encoding="utf-8") as f:
    records = [r for r in json.load(f) if r["tokens"]]

dx = [[t[0] for t in r["tokens"]] for r in records]
dy = [[t[3] for t in r["tokens"]] for r in records]
pred = model.predict(dx)

print(classification_report(dy, pred, digits=3))

groups = defaultdict(lambda: ([], []))
for r, g, p in zip(records, dy, pred):
    s = source_of[r["uid"]]
    groups[s][0].append(g)
    groups[s][1].append(p)

print("By source:")
for s, (g, p) in sorted(groups.items()):
    n_ents = sum(1 for seq in g for t in seq if t.startswith("B-"))
    print(f"  {s:12}{len(g):6} sentences  {n_ents:5} entities  F1 {f1_score(g, p):.3f}")