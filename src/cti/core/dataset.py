import json
import os
from pathlib import Path

import pandas as pd


def load_smah(raw_dir="data/raw", splits_dir="data/splits", splits=None):
    """Return split dataframes using the official report-level split.

    Refuses to load the test split unless CTI_ALLOW_TEST=1 is set in the environment.
    """
    allow_test = os.environ.get("CTI_ALLOW_TEST") == "1"

    if splits is None:
        splits = ["train", "dev", "test"] if allow_test else ["train", "dev"]
    elif "test" in splits and not allow_test:
        raise PermissionError(
            "Access to the test split is locked. "
            "Set environment variable CTI_ALLOW_TEST=1 to allow loading test data."
        )

    path = next(Path(raw_dir).rglob("SMAH-CTI.json"))
    with open(path, encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))

    parts = {}
    for name in splits:
        with open(Path(splits_dir) / f"{name}_report_ids.json", encoding="utf-8") as f:
            ids = set(json.load(f))
        parts[name] = df[df["doc_id"].isin(ids)].reset_index(drop=True)
    return parts
