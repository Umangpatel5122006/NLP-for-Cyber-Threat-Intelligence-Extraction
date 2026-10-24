import json
from pathlib import Path

import pandas as pd


def load_smah(raw_dir="data/raw", splits_dir="data/splits"):
    """Return {'train': df, 'dev': df, 'test': df} using the official split."""
    path = next(Path(raw_dir).rglob("SMAH-CTI.json"))
    with open(path, encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))

    parts = {}
    for name in ["train", "dev", "test"]:
        with open(Path(splits_dir) / f"{name}_report_ids.json", encoding="utf-8") as f:
            ids = set(json.load(f))
        parts[name] = df[df["doc_id"].isin(ids)].reset_index(drop=True)
    return parts