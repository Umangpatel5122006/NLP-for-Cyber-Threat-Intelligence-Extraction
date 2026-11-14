"""Train/save sentence relevance on train; evaluate only the official dev split."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from cti.core.dataset import load_smah
from cti.models.relevance import SentenceRelevanceClassifier


ROOT = Path(__file__).resolve().parents[2]


def _training_rows(frame):
    if frame["uid"].duplicated().any():
        raise ValueError("Sentence UIDs must be unique within a split.")
    values = [str(value).strip().casefold() for value in frame["relevant"]]
    if any(value not in ("yes", "no") for value in values):
        raise ValueError("Relevance labels must be 'yes' or 'no'.")
    return frame["sentence_text"].tolist(), [int(value == "yes") for value in values]


def train_and_evaluate(raw_dir, splits_dir, output, report_path, threshold=0.5):
    # Explicit selection keeps this command train/dev-only even if a caller has
    # CTI_ALLOW_TEST set for unrelated work in the same shell.
    splits = load_smah(raw_dir=raw_dir, splits_dir=splits_dir, splits=["train", "dev"])
    train, dev = splits["train"], splits["dev"]
    if set(train["doc_id"]) & set(dev["doc_id"]) or set(train["uid"]) & set(dev["uid"]):
        raise ValueError("Train and dev must not share reports or sentence UIDs.")
    protected_ids = set(json.loads(
        (Path(splits_dir) / "test_report_ids.json").read_text(encoding="utf-8")
    ))
    if (set(train["doc_id"]) | set(dev["doc_id"])) & protected_ids:
        raise PermissionError("Protected test reports must not appear in train or dev.")
    train_texts, train_labels = _training_rows(train)
    dev_texts, dev_labels = _training_rows(dev)
    if not dev_texts:
        raise ValueError("The development split must contain sentences.")

    model = SentenceRelevanceClassifier(threshold=threshold).fit(train_texts, train_labels)
    scores = model.predict_scores(dev_texts)
    predicted = [int(score >= model.threshold) for score in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(
        dev_labels, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(dev_labels, predicted, labels=[0, 1]).ravel()
    metrics = {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }
    raw_path = next(Path(raw_dir).rglob("SMAH-CTI.json"))
    code_files = [
        "src/cti/train_relevance.py", "src/cti/models/relevance.py",
        "src/cti/preprocessing/cleaner.py", "src/cti/preprocessing/defanger.py",
    ]
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("scikit-learn", "numpy", "pandas", "joblib")},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code_files},
        "raw_dataset_sha256_lf": hashlib.sha256(raw_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "train_report_ids": sorted(set(train["doc_id"])),
        "dev_report_ids": sorted(set(dev["doc_id"])),
        "training_sentences": len(train_texts),
        "training_relevant": sum(train_labels),
        "dev_sentences": len(dev_texts),
        "dev_relevant": sum(dev_labels),
        "evaluation_scope": "official dev sentences; dataset-provided sentence boundaries",
        "metric": "binary precision/recall/F1 for the relevant class",
        "dev_metrics": metrics,
        "test_split_used": False,
    }
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    provenance["base_commit"] = git_result.stdout.strip() if git_result.returncode == 0 else None

    output, report_path = Path(output), Path(report_path)
    predictions_path = report_path.with_suffix(".predictions.csv")
    paths = [output, output.with_suffix(".manifest.json"), report_path, predictions_path]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Model, manifest, report, and predictions paths must be distinct.")
    manifest = model.save(output, metadata=provenance)
    loaded = SentenceRelevanceClassifier.load(output)
    if loaded.predict(dev_texts) != [bool(value) for value in predicted]:
        raise RuntimeError("Saved-model reload changed dev predictions.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "uid", "doc_id", "gold_relevant", "predicted_relevant", "relevance_score"
        ])
        writer.writeheader()
        for uid, doc_id, gold, prediction, score in zip(
            dev["uid"], dev["doc_id"], dev_labels, predicted, scores
        ):
            writer.writerow({
                "uid": uid, "doc_id": doc_id, "gold_relevant": gold,
                "predicted_relevant": prediction, "relevance_score": score,
            })
    report = {
        **manifest,
        "artifact_file": output.as_posix(),
        "predictions_file": predictions_path.as_posix(),
        "reload_predictions_identical": True,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--output", type=Path, default=Path("models/relevance.joblib"))
    parser.add_argument("--report", type=Path, default=Path("results/relevance_v1_dev.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    report = train_and_evaluate(
        args.raw_dir, args.splits_dir, args.output, args.report, args.threshold
    )
    metrics = report["metadata"]["dev_metrics"]
    print(f"Dev relevance: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
    print(f"Saved model: {args.output}; dev report: {args.report}")
    print("Reload predictions identical; test split not used.")


if __name__ == "__main__":
    main()
