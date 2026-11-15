"""Train-only grouped classical SMS comparison; dev evaluation; no test access."""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from cti.core.sms_data import DATA, SPLITS, create_split, download_sms, load_sms
from cti.models.message_classifier import MessageClassifier, PROFILES, new_pipeline


def metrics(gold, predictions):
    p, r, f1, _ = precision_recall_fscore_support(gold, predictions, average="binary", pos_label="spam", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(gold, predictions, labels=["ham", "spam"]).ravel()
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def train(data=DATA, splits=SPLITS, output=Path("models/message_spam.joblib"), results=Path("results/message_model_v1")):
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    train_rows = load_sms("train", data, splits)
    texts = [r["text"] for r in train_rows]
    labels = np.array([r["label"] for r in train_rows])
    groups = [r["group"] for r in train_rows]
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(texts, labels, groups))
    scores = []
    warnings.filterwarnings("error", category=ConvergenceWarning)
    for profile in PROFILES:
        pooled = [None] * len(texts)
        fold_scores = []
        for fold, (fit_ids, validation_ids) in enumerate(folds):
            assert set(groups[i] for i in fit_ids).isdisjoint(groups[i] for i in validation_ids)
            model = new_pipeline(profile)
            model.fit([texts[i] for i in fit_ids], labels[fit_ids])
            predictions = model.predict([texts[i] for i in validation_ids]).tolist()
            for i, prediction in zip(validation_ids, predictions):
                pooled[i] = prediction
            fold_scores.append(metrics(labels[validation_ids], predictions))
            print(f"{profile} fold {fold + 1}: {fold_scores[-1]}", flush=True)
        scores.append({"profile": profile, "pooled_train_cv": metrics(labels, pooled), "folds": fold_scores})
    selected = max(scores, key=lambda item: (item["pooled_train_cv"]["f1"], item["pooled_train_cv"]["precision"], -PROFILES.index(item["profile"])))
    selection = {"selected": selected["profile"], "selection_rule": "max pooled training CV spam F1, precision then fixed profile order",
                 "candidates": scores, "dev_used_for_selection": False, "test_evaluated": False,
                 "threshold": 0.0, "train_rows": len(texts), "train_groups": len(set(groups)),
                 "dataset_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
                 "split_sha256": hashlib.sha256(Path(splits).read_bytes()).hexdigest()}
    (results / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print("Selection frozen before loading dev:", selected["profile"], flush=True)
    model = new_pipeline(selected["profile"])
    model.fit(texts, labels)
    dev = load_sms("dev", data, splits)
    predictions = model.predict([r["text"] for r in dev]).tolist()
    report = {"task": "sms_ham_spam", "profile": selected["profile"], "split": "dev", "rows": len(dev),
              "groups": len({r["group"] for r in dev}), "metrics": metrics([r["label"] for r in dev], predictions),
              "test_evaluated": False, "selection": selection,
              "limitations": "UCI SMS ham/spam, not universal scam/attack accuracy. Template-grouped dev, not final test."}
    saved = MessageClassifier.save(model, output, selected["profile"], {"selection": selection, "dev_metrics": report["metrics"]})
    restored = MessageClassifier.load(output)
    assert restored.pipeline.predict([r["text"] for r in dev]).tolist() == predictions
    report["artifact_sha256"] = saved.manifest["artifact_sha256"]
    report["reload_predictions_identical"] = True
    (results / "dev.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (results / "dev_predictions.json").write_text(json.dumps([
        {"id": row["id"], "gold": row["label"], "prediction": prediction} for row, prediction in zip(dev, predictions)
    ], indent=2) + "\n", encoding="utf-8")
    print("Dev:", report["metrics"], flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--output", type=Path, default=Path("models/message_spam.joblib"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/message_model_v1"))
    args = parser.parse_args(argv)
    if args.download:
        download_sms(args.data)
    if not args.data.is_file():
        parser.error("SMS data missing; run with --download to fetch the pinned official corpus.")
    if not args.splits.exists():
        create_split(args.data, args.splits)
    train(args.data, args.splits, args.output, args.results_dir)


if __name__ == "__main__":
    main()
