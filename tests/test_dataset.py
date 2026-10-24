from cti.core.dataset import load_smah

parts = load_smah()
for name, part in parts.items():
    print(name, len(part), "sentences,", part["doc_id"].nunique(), "reports")