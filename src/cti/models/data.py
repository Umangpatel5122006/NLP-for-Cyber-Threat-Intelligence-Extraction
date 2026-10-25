import json
from pathlib import Path


def load_bio(layer, split, lang="en", root="data/processed"):
    with open(Path(root) / f"bio_{lang}_{layer}_{split}.json", encoding="utf-8") as f:
        records = json.load(f)
    tokens, tags = [], []
    for r in records:
        if r["tokens"]:
            tokens.append([t[0] for t in r["tokens"]])
            tags.append([t[3] for t in r["tokens"]])
    return tokens, tags