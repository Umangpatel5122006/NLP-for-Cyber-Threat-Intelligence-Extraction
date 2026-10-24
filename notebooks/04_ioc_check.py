import cti.extractors.ioc_finder
import cti.preprocessing.defanger
from cti.core.base import Document
from cti.core.dataset import load_smah
from cti.core.registry import create

dev = load_smah()["dev"]
dev = dev[dev["entities"].notna()]

defang = create("defanger")
finder = create("ioc_finder")


def fix(text):
    return defang.process(Document(text=text)).text.strip().lower()


total, hit = 0, 0
for _, row in dev.iterrows():
    gold = [e for e in row["entities"] if e["label"] == "Infrastructure_Indicator"]
    if not gold:
        continue
    doc = defang.process(Document(text=row["sentence_text"]))
    found = {e.text.lower() for e in finder.process(doc).entities}
    for g in gold:
        total += 1
        if fix(g["text"]) in found:
            hit += 1

print(f"Dataset indicator spans: {total}")
print(f"Caught by defanger + ioc_finder: {hit} ({100*hit/total:.1f}%)")