
"""
give all unique types can appear in training dataset of 
Threat Actor , Action , Infrastructure_Indicator , MalwareTool

"""



import random
from collections import defaultdict

from cti.core.dataset import load_smah

train = load_smah()["train"]

by_label = defaultdict(list)
for ents in train["entities"]:
    if ents:
        for e in ents:
            by_label[e["label"]].append(e["text"])

random.seed(1)
for label, texts in by_label.items():
    print(f"\n{label} ({len(texts)} total)")
    for t in random.sample(texts, 12):
        print("  -", t)