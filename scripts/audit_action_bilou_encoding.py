"""Verify train-only BILOU encoding without fitting or importing a tokenizer."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from cti.core.dataset import load_smah
from cti.core.sms_data import ROOT
from cti.models.action_bilou import bilou_is_valid, bilou_to_bio, bio_to_bilou
from cti.models.ner_artifact import sha256_file
from cti.models.ner_data import load_aligned_ner


def main():
    previous = json.loads((ROOT / "results/action_background_v1/summary.json").read_text(encoding="utf-8"))["provenance"]
    unchanged = dict(previous["unchanged_files_sha256"])
    unchanged["configs/config_ner.yaml"] = hashlib.sha256((ROOT / "configs/config_ner.yaml").read_bytes()).hexdigest()
    for name, expected in unchanged.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("An active artifact/configuration differs from the baseline: " + name)
    paths = {"raw": ROOT / "data/raw/SMAH-CTI.json", "processed_train": ROOT / "data/processed/bio_en_action_train.json"}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes["raw"] != previous["raw_sha256_lf"] or hashes["processed_train"] != previous["processed_train_sha256_lf"]:
        raise ValueError("Frozen training inputs changed.")
    for split, expected in previous["split_sha256_lf"].items():
        if sha256_file(ROOT / f"data/splits/{split}_report_ids.json") != expected:
            raise ValueError("Official split definitions changed.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    train = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    counts, gold_spans, tokens = Counter(), 0, 0
    for words, gold in zip(train.tokens, train.tags):
        encoded = bio_to_bilou(gold)
        if not bilou_is_valid(encoded) or bilou_to_bio(encoded) != gold or len(encoded) != len(words):
            raise ValueError("Encoding changed a gold token or action boundary.")
        gold_spans += sum(tag == "B-ACTION" for tag in gold)
        tokens += len(words)
        counts.update(encoded)
    if gold_spans != counts["L-ACTION"] + counts["U-ACTION"]:
        raise ValueError("BILOU end/unit counts do not conserve gold action spans.")
    if hashes != {name: sha256_file(path) for name, path in paths.items()}:
        raise ValueError("Training files changed during the audit.")
    output = ROOT / "results/action_bilou_v1/preparation"
    output.mkdir(parents=True, exist_ok=True)
    result = {"passed": True, "scope": "Gold train-label encoding only; no model training or accuracy evaluation.",
              "training_reports": len(set(train.doc_ids)), "training_sentences": len(train.uids), "training_tokens": tokens,
              "gold_action_spans": gold_spans, "internal_tag_counts": dict(sorted(counts.items())), "gold_bio_roundtrip_identical": True,
              "active_artifacts_and_configs_unchanged": True, "unchanged_files_sha256": unchanged,
              "dev_evaluated": False, "test_evaluated": False, "model_training_performed": False,
              "data_sha256_lf": hashes, "source_sha256_lf": {name: sha256_file(ROOT / name) for name in (
                  "scripts/audit_action_bilou_encoding.py", "src/cti/models/action_bilou.py", "src/cti/models/ner_data.py")}}
    (output / "integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
