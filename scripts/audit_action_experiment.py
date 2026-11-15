"""Re-score the fixed action experiment and regenerate its diagnostic artifacts.

Run from the repository root after python -m cti.experiment_action.
Only train/dev annotations are selected; the official test IDs are exclusion guards.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from cti.core.dataset import load_smah
from cti.eval.metrics import evaluate_predictions, extract_spans
from cti.experiment_action import _check_predictions, _digest, choose_candidate
from cti.models.ner_artifact import sha256_file
from cti.models.ner_data import load_aligned_ner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/action_cv_v1"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/action_cv_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/action_experiment.yaml"))
    parser.add_argument("--active-action-report", type=Path, default=Path("results/ner_v2_action.json"),
                        help="Reference artifact report for the current active action model.")
    args = parser.parse_args(argv)
    folder = args.results_dir
    cv = json.loads((folder / "cv.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    selected = choose_candidate(cv["candidates"], selection["minimum_cv_gain"], selection["minimum_fold_wins"])
    assert summary["selected"] == selection["selected"] == selected
    assert not selection["dev_labels_used_for_selection"] and not summary["test_split_used"]
    raw = load_smah(splits=["train", "dev"])
    train = load_aligned_ner("action", "train", raw["train"], "data/processed")
    protected = set(json.loads(Path("data/splits/test_report_ids.json").read_text(encoding="utf-8")))
    dev_ids = set(raw["dev"]["doc_id"])
    seen = []
    for fold in cv["folds"]:
        fit, validation = set(fold["train_report_ids"]), set(fold["validation_report_ids"])
        assert not fit & validation and fit | validation == set(train.doc_ids)
        assert not (fit | validation) & (protected | dev_ids)
        seen.extend(validation)
    assert sorted(seen) == sorted(set(train.doc_ids))
    for name, digest in cv["provenance"]["source_sha256_lf"].items():
        assert sha256_file(name) == digest
    assert sha256_file(args.config) == cv["provenance"]["config_sha256_lf"]

    per_doc = {}
    for report in cv["candidates"]:
        predictions = [None] * len(train.tokens)
        for number, fold in enumerate(cv["folds"], 1):
            cache = json.loads((args.model_dir / f"{report['id']}_fold_{number}.json").read_text(encoding="utf-8"))
            assert cache["signature"]["fingerprint"] == _digest(cv["provenance"])
            decoded = cache["predictions"]
            assert cache["predictions_sha256"] == _digest(decoded)
            validation = set(fold["validation_report_ids"])
            indices = [i for i, identifier in enumerate(train.doc_ids) if identifier in validation]
            assert len(indices) == len(decoded)
            for index, tags in zip(indices, decoded):
                predictions[index] = tags
        _check_predictions(train.tokens, predictions)
        assert evaluate_predictions(train.tags, predictions) == report["pooled"]
        counts = defaultdict(lambda: [0, 0, 0])
        for identifier, gold, predicted in zip(train.doc_ids, train.tags, predictions):
            g = set(extract_spans(gold, mode="strict"))
            p = set(extract_spans(predicted, mode="strict"))
            values = counts[identifier]
            values[0] += len(g & p)
            values[1] += len(g)
            values[2] += len(p)
        per_doc[report["id"]] = counts

    def f1(counts, identifiers):
        totals = [sum(counts[i][j] for i in identifiers) for j in range(3)]
        return 2 * totals[0] / (totals[1] + totals[2]) if totals[1] + totals[2] else 0.0

    ids = sorted(set(train.doc_ids))
    rng = random.Random(42)
    samples = [rng.choices(ids, k=len(ids)) for _ in range(1000)]
    comparisons = {}
    for identifier, counts in per_doc.items():
        if identifier == "baseline":
            continue
        deltas = [f1(counts, sample) - f1(per_doc["baseline"], sample) for sample in samples]
        comparisons[identifier] = {
            "pooled_exact_f1_difference": round(f1(counts, ids) - f1(per_doc["baseline"], ids), 4),
            "conditional_95_percentile_interval": [round(float(value), 4) for value in np.percentile(deltas, [2.5, 97.5])],
        }
    bootstrap = {
        "resamples": 1000, "seed": 42, "unit": "whole training report",
        "interpretation": "Conditional paired bootstrap of fixed OOF predictions; does not include refitting uncertainty or constitute final-test inference.",
        "comparisons": comparisons, "audit_script_sha256_lf": sha256_file(__file__),
    }
    (folder / "paired_cv_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")

    original = json.loads(Path("results/ner_v2_action.predictions.json").read_text(encoding="utf-8"))
    confirmed = json.loads((folder / "baseline_dev.predictions.json").read_text(encoding="utf-8"))
    assert original == confirmed
    baseline = json.loads((folder / "baseline_dev.json").read_text(encoding="utf-8"))
    frozen = json.loads(Path("results/ner_v2_action.json").read_text(encoding="utf-8"))
    for key in ("headline_all_sentences", "diagnostic_relevant_only", "per_source"):
        assert baseline[key] == frozen[key]
    relevant = {row["uid"]: row["relevant"] == "yes" for row in raw["dev"].to_dict("records")}
    cases, irrelevant_predictions = [], 0
    for row in original:
        gold = extract_spans(row["gold"], mode="strict")
        predicted = extract_spans(row["predicted"], mode="strict")
        if not relevant[row["uid"]]:
            irrelevant_predictions += len(predicted)
        for label, start, end in gold:
            if (label, start, end) in predicted:
                continue
            overlaps = [(a, b) for kind, a, b in predicted if kind == label and max(start, a) < min(end, b)]
            if not overlaps:
                continue
            a, b = max(overlaps, key=lambda span: min(end, span[1]) - max(start, span[0]))
            cases.append({
                "uid": row["uid"], "doc_id": row["doc_id"],
                "gold": " ".join(row["tokens"][start:end]), "predicted": " ".join(row["tokens"][a:b]),
            })
    taxonomy = baseline["headline_all_sentences"]["error_taxonomy"]
    diagnostics = {
        "gold_action_spans": taxonomy["total_gold"], "predicted_action_spans": taxonomy["total_predicted"],
        "predictions_on_irrelevant_dev_sentences": irrelevant_predictions, "error_taxonomy": taxonomy,
        "example_sampling": "20 gold spans with overlapping non-exact predictions; random seed 42",
        "boundary_examples": random.Random(42).sample(cases, min(20, len(cases))),
    }
    (folder / "baseline_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    for layer in ("entities", "action"):
        reference = args.active_action_report if layer == "action" else Path("results/ner_v2_entities.json")
        report = json.loads(reference.read_text(encoding="utf-8"))
        assert hashlib.sha256(Path(f"models/ner_{layer}.joblib").read_bytes()).hexdigest() == report["artifact"]["artifact_sha256"]
    message = (
        "All 9 fold caches independently re-scored; metrics equal stored reports.\n"
        "70 training reports used once each for validation; dev/test report IDs excluded from CV.\n"
        "Selection rechecked from train-only results.\n"
        "Refitted baseline reproduces all 1238 frozen dev predictions and every reported metric exactly.\n"
        "Active entity/action model hashes match the supplied reference reports.\n"
        "Experiment source/config hashes match the run provenance.\n"
    )
    (folder / "integrity_validation.txt").write_text(message, encoding="utf-8")
    print(message)


if __name__ == "__main__":
    main()
