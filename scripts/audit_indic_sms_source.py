"""Audit a pinned small Indian SMS source; never fit or evaluate a model."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from cti.core.sms_data import ROOT
from cti.core.sms_domain import audit_overlap
from cti.preprocessing.language_detector import describe_language
from cti.sms_intake import protected_references


URL = "https://www.kaggle.com/api/v1/datasets/download/uds5501/sms-dataset?datasetVersionNumber=1"
ARCHIVE_SHA = "b1ed0eaccff68ee427855276c4d50fdac14cc3be328c89d880bcdb1320465cb8"
CSV_SHA = "a1c138cd6febb8bebb799f8a5242a3cb8f95bd66cb5f1876a82b648217ea4447"
DATA = ROOT / "models/indic_research/kaggle_sms_v1.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    if args.download:
        with urlopen(URL, timeout=40) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000 or hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA:
            raise ValueError("Indian SMS archive changed; review before use.")
        with ZipFile(io.BytesIO(payload)) as archive:
            if archive.getinfo("new.csv").file_size > 300_000:
                raise ValueError("Unexpected source member size.")
            raw = archive.read("new.csv")
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_bytes(raw)
    raw = DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CSV_SHA:
        raise ValueError("Pinned Indian SMS source changed.")
    parsed = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(parsed) != 184 or Counter(r["Label"] for r in parsed) != {"ham": 77, "spam": 107}:
        raise ValueError("Source schema/counts differ from the reviewed version.")
    rows = [{"id": f"indian-sms-{i:03d}", "text": row["Message"], "original_label": row["Label"]}
            for i, row in enumerate(parsed)]
    texts, protected_hashes = protected_references()
    retained, overlap = audit_overlap(rows, texts)
    report = {"source": "uds5501 / SMS dataset / Kaggle v1 (2018)",
              "source_url": "https://www.kaggle.com/datasets/uds5501/sms-dataset", "license_as_declared": "CC BY-SA 3.0",
              "archive_sha256": ARCHIVE_SHA, "csv_sha256": CSV_SHA, "source_labels": dict(Counter(r["original_label"] for r in rows)),
              "devanagari_rows": sum(any("\u0900" <= c <= "\u097f" for c in r["text"]) for r in rows),
              "overlap_audit": overlap,
              "retained_unverified_language_guesses": dict(Counter(describe_language(r["text"])["language"] for r in retained)),
              "language_gold_available": False, "trained_on_source": False, "test_evaluated": False,
              "decision": "Small source for future review, not Hindi/Hinglish model validation. No Devanagari Hindi; no author language labels or independent adjudication records.",
              "protected_source_sha256": protected_hashes}
    output = ROOT / "results/indic_integration_v1/model_scope_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("source_labels", "devanagari_rows", "overlap_audit", "retained_unverified_language_guesses", "decision")}, indent=2))


if __name__ == "__main__":
    main()
