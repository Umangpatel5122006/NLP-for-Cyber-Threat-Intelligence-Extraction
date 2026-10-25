
"""
label                gold   kept   lost %
THREAT_ACTOR         1144   1107     3.2%
ACTION               4470   4437     0.7%
INFRASTRUCTURE       2998   2476    17.4%
MALWARE_TOOL         1815   1452    20.0%

"""


from collections import Counter

from cti.core.bio import load_label_map, to_bio
from cti.core.dataset import load_smah

label_map = load_label_map()
train = load_smah()["train"]

gold, kept = Counter(), Counter()
for _, row in train.iterrows():
    ents = row["entities"] or []
    gold.update(label_map[e["label"]] for e in ents)
    bio = to_bio(row["sentence_text"], ents, label_map)
    kept.update(t[2:] for _, _, _, t in bio if t.startswith("B-"))

print(f"{'label':18}{'gold':>7}{'kept':>7}{'lost %':>9}")
for label in gold:
    lost = 100 * (gold[label] - kept[label]) / gold[label]
    print(f"{label:18}{gold[label]:>7}{kept[label]:>7}{lost:>8.1f}%")