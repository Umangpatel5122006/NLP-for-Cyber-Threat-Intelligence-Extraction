"""
Convert whole dataset into bio tags.

we convert all 12,304 and save the result, so the CRF training in next step can load it directly. We’ll also count 
how many dataset spans survive, because filter_spans drops overlapping ones, and we should know how many it dropped.




Before Entity and Action Layer seperate out Result:

Split	Gold spans	    Kept	Lost
Train	10,427	        9,472	~9%
Dev	    1,520	        1,318	~13%
Test	3,540	        3,164	~11%

After separating Entity and Action Layer Result:

train entities   gold:  5957  kept:  5914  lost: 0.7%
train action     gold:  4470  kept:  4453  lost: 0.4%
dev   entities   gold:   912  kept:   903  lost: 1.0%
dev   action     gold:   608  kept:   608  lost: 0.0%
test  entities   gold:  2072  kept:  2054  lost: 0.9%
test  action     gold:  1468  kept:  1453  lost: 1.0%

"""
import json
from pathlib import Path

from cti.core.bio import load_label_map, load_layers, to_bio
from cti.core.dataset import load_smah

LANG = "en"        # SMAH-CTI is English; a Hindi dataset would use "hi"

layers = load_layers()
label_map = load_label_map()
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

for split, df in load_smah().items():
    for layer, keep in layers.items():
        gold, kept, records = 0, 0, []
        for _, row in df.iterrows():
            ents = row["entities"] or []
            gold += sum(1 for e in ents if label_map[e["label"]] in keep)
            bio = to_bio(row["sentence_text"], ents, label_map, lang=LANG, keep=keep)
            kept += sum(1 for _, _, _, t in bio if t.startswith("B-"))
            records.append({"uid": row["uid"], "tokens": bio})

        with open(out_dir / f"bio_{LANG}_{layer}_{split}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        loss = 100 * (gold - kept) / gold if gold else 0
        print(f"{split:6}{layer:10} gold: {gold:5}  kept: {kept:5}  lost: {loss:.1f}%")