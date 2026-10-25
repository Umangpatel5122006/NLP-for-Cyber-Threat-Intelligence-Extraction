import importlib
import time

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from cti.core.registry import create
from cti.models.data import load_bio

LAYER = "entities"      # or "action"
LANG = "en"

# registry name -> file that defines it. Add or remove models here.
MODELS = {
    # "crf_tagger": "cti.models.crf",
    # "spacy_ner": "cti.models.spacy_ner",
    # "bilstm_crf": "cti.models.bilstm_crf",
    "bert_ner": "cti.models.bert_ner",
}

train_x, train_y = load_bio(LAYER, "train", LANG)
dev_x, dev_y = load_bio(LAYER, "dev", LANG)

rows = []
for name, module in MODELS.items():
    importlib.import_module(module)
    model = create(name)
    print(f"\n=== {name} on layer '{LAYER}' ===")
    t0 = time.time()
    model.fit(train_x, train_y, dev_x, dev_y)
    train_time = time.time() - t0
    pred = model.predict(dev_x)
    print(classification_report(dev_y, pred, digits=3))
    rows.append((name, precision_score(dev_y, pred), recall_score(dev_y, pred),
                 f1_score(dev_y, pred), train_time))

print("\nSUMMARY (dev set, exact entity match)")
print(f"{'model':14}{'precision':>10}{'recall':>9}{'F1':>8}{'train s':>9}")
for name, p, r, f, t in rows:
    print(f"{name:14}{p:>10.3f}{r:>9.3f}{f:>8.3f}{t:>9.0f}")