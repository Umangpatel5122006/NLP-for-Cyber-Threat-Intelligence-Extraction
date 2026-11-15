"""Source-aware learned SMS comparison. Select on train CV; gate on dev; no tests."""

import argparse
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

from cti.core.sms_data import DATA, ROOT, SPLITS, download_sms, load_sms
from cti.core.sms_domain import SOURCE, SPLIT, download_domain, load_domain, prepare_domain
from cti.models.message_classifier import MessageClassifier, new_pipeline
from cti.models.message_v2 import ExpandedMessageClassifier, load_message_classifier, new_expanded_pipeline
from cti.train_message import metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def original_rows(split):
    return [{**r, "source": "uci", "original_label": r["label"]} for r in load_sms(split)]


def score_sources(rows, predictions):
    per_source = {}
    for source in ("uci", "mendeley"):
        indices = [i for i, r in enumerate(rows) if r["source"] == source]
        per_source[source] = metrics([rows[i]["label"] for i in indices], [predictions[i] for i in indices])
    return {"per_source": per_source,
            "source_mean_f1": sum(m["f1"] for m in per_source.values()) / len(per_source)}


def choose(cv, policy):
    base = cv[0]
    eligible = []
    for candidate in cv[1:]:
        gains = [a["source_mean_f1"] > b["source_mean_f1"] for a, b in zip(candidate["folds"], base["folds"])]
        good = (candidate["pooled"]["source_mean_f1"] >= base["pooled"]["source_mean_f1"] + policy["minimum_source_mean_f1_gain"]
                and sum(gains) >= policy["minimum_fold_wins"]
                and candidate["pooled"]["per_source"]["uci"]["f1"] >= base["pooled"]["per_source"]["uci"]["f1"] - policy["maximum_uci_f1_loss"]
                and candidate["pooled"]["per_source"]["uci"]["fp"] <= base["pooled"]["per_source"]["uci"]["fp"] + policy["maximum_extra_uci_false_positives"])
        candidate["eligible"] = bool(good)
        candidate["fold_wins"] = sum(gains)
        if good:
            eligible.append(candidate)
    return max(eligible, key=lambda c: c["pooled"]["source_mean_f1"])["id"] if eligible else base["id"]


def promotion_checks(base, candidate, policy):
    return {
        "uci_f1": candidate["per_source"]["uci"]["f1"] >= base["per_source"]["uci"]["f1"] - policy["maximum_uci_dev_f1_loss"],
        "uci_false_positives": candidate["per_source"]["uci"]["fp"] <= base["per_source"]["uci"]["fp"] + policy["maximum_extra_uci_dev_false_positives"],
        "domain_f1": candidate["per_source"]["mendeley"]["f1"] >= base["per_source"]["mendeley"]["f1"] + policy["minimum_domain_dev_f1_gain"],
        "source_mean_f1": candidate["source_mean_f1"] >= base["source_mean_f1"] + policy["minimum_dev_source_mean_f1_gain"],
    }


def factory(profile):
    return new_pipeline(profile) if profile == "word_char_svm" else new_expanded_pipeline(profile)


def fit(candidate, rows):
    selected = rows if candidate["training"] == "all" else [r for r in rows if r["source"] == "uci"]
    model = factory(candidate["profile"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit([r["text"] for r in selected], [r["label"] for r in selected])
    return model


def run(results_dir=ROOT / "results/message_generalization_v1", model_dir=ROOT / "models/message_generalization_v1",
        config_path=ROOT / "configs/message_expansion.yaml"):
    results_dir, model_dir = Path(results_dir), Path(model_dir)
    results_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dependencies = ["src/cti/core/sms_data.py", "src/cti/core/sms_domain.py", "src/cti/models/message_classifier.py",
                    "src/cti/models/message_features.py", "src/cti/models/message_v2.py", "src/cti/train_message.py", "src/cti/experiment_message.py"]
    provenance = {"config_sha256": digest(config_path), "source_sha256": {p: digest(ROOT / p) for p in dependencies},
                  "uci_split_sha256": digest(SPLITS), "domain_split_sha256": digest(SPLIT)}
    active = ROOT / "models/message_spam.joblib"
    active_sha = hashlib.sha256(active.read_bytes()).hexdigest()
    baseline_artifact = load_message_classifier(active)
    train = original_rows("train") + load_domain("train")
    folds = list(StratifiedGroupKFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"]).split(
        train, [r["source"] + ":" + r["original_label"] for r in train], [r["group"] for r in train]))
    cv = []
    for candidate in config["candidates"]:
        pooled, fold_metrics = [None] * len(train), []
        for number, (train_ids, valid_ids) in enumerate(folds, 1):
            assert {train[i]["group"] for i in train_ids}.isdisjoint(train[i]["group"] for i in valid_ids)
            fitted = fit(candidate, [train[i] for i in train_ids])
            predicted = fitted.predict([train[i]["text"] for i in valid_ids]).tolist()
            for i, label in zip(valid_ids, predicted):
                pooled[i] = label
            fold_metrics.append(score_sources([train[i] for i in valid_ids], predicted))
            print(candidate["id"], "fold", number, fold_metrics[-1], flush=True)
        cv.append({"id": candidate["id"], "profile": candidate["profile"], "pooled": score_sources(train, pooled), "folds": fold_metrics})
        write(results_dir / (candidate["id"] + "_oof.json"), [
            {"id": r["id"], "source": r["source"], "group": r["group"], "gold": r["label"], "predicted": p}
            for r, p in zip(train, pooled)])
    selected_id = choose(cv, config["selection"])
    selection = {"selected": selected_id, "policy": config["selection"], "candidates": cv,
                 "dev_used_for_selection": False, "test_evaluated": False, "provenance": provenance}
    write(results_dir / "selection.json", selection)
    print("Selection frozen before loading dev:", selected_id, flush=True)
    selected = next(c for c in config["candidates"] if c["id"] == selected_id)
    training_rows = sum(selected["training"] == "all" or r["source"] == "uci" for r in train)
    fitted = fit(selected, train)
    dev = original_rows("dev") + load_domain("dev")
    texts = [r["text"] for r in dev]
    base_predictions = baseline_artifact.pipeline.predict(texts).tolist()
    predictions = fitted.predict(texts).tolist()
    base_scores, scores = score_sources(dev, base_predictions), score_sources(dev, predictions)
    checks = promotion_checks(base_scores, scores, config["promotion"])
    output = model_dir / "candidate.joblib"
    saver = MessageClassifier if selected["profile"] == "word_char_svm" else ExpandedMessageClassifier
    saved = saver.save(fitted, output, selected["profile"], {
        "training_protocol": "expanded_smishing_v1", "selection": selection, "dev_metrics": scores["per_source"]["uci"],
        "domain_dev_metrics": scores["per_source"]["mendeley"], "training_rows": training_rows,
        "training_sources": ["uci", "mendeley"] if selected["training"] == "all" else ["uci"]})
    restored = load_message_classifier(output)
    assert restored.pipeline.predict(texts).tolist() == predictions
    assert provenance["source_sha256"] == {p: digest(ROOT / p) for p in dependencies}
    assert hashlib.sha256(active.read_bytes()).hexdigest() == active_sha
    report = {"baseline": base_scores, "candidate": scores, "promotion_checks": checks,
              "rows_by_source": {s: sum(r["source"] == s for r in dev) for s in ("uci", "mendeley")},
              "smishing_recall": sum(p == "spam" for r, p in zip(dev, predictions) if r["original_label"] == "smishing") / sum(r["original_label"] == "smishing" for r in dev),
              "reload_predictions_identical": True, "test_evaluated": False}
    write(results_dir / "dev.json", report)
    write(results_dir / "dev_predictions.json", [{"id": r["id"], "source": r["source"], "gold": r["label"], "baseline": b, "candidate": p}
          for r, b, p in zip(dev, base_predictions, predictions)])
    # The authored challenge is diagnostic only, read after train-only selection/dev gating.
    challenge = json.loads((ROOT / "examples/message_challenge.json").read_text(encoding="utf-8"))
    cases = challenge["messages"]
    old = baseline_artifact.pipeline.predict([r["text"] for r in cases]).tolist()
    new = fitted.predict([r["text"] for r in cases]).tolist()
    challenge_report = {"origin": challenge["origin"], "used_for_selection": False, "limitations": challenge["limitations"],
                        "baseline": metrics([r["label"] for r in cases], old), "candidate": metrics([r["label"] for r in cases], new),
                        "cases": [{**r, "baseline": a, "candidate": b} for r, a, b in zip(cases, old, new)]}
    write(results_dir / "challenge.json", challenge_report)
    probe = "PM Modi is giving 15 lakhs rupees to every citizen of our country. To credit it in your bank account. Kindly SMS 6969."
    summary = {"selected": selected_id, "profile": selected["profile"], "training_rows": training_rows,
               "baseline_dev": base_scores, "candidate_dev": scores, "promotion_checks": checks,
               "eligible_for_activation": selected_id != "uci_reference" and all(checks.values()),
               "candidate_path": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
               "candidate_sha256": saved.manifest["artifact_sha256"], "previous_active_sha256": active_sha,
               "challenge_baseline": challenge_report["baseline"], "challenge_candidate": challenge_report["candidate"],
               "known_user_probe": {"used_for_selection": False, "baseline": baseline_artifact.predict_one(probe), "candidate": restored.predict_one(probe)},
               "deployment_hold": "authored_diagnostic_false_alarms_increased" if challenge_report["candidate"]["fp"] > challenge_report["baseline"]["fp"] else None,
               "hold_interpretation": "Post-experiment release-readiness finding, not a change to CV selection or threshold; challenge data were not fitted.",
               "active_model_modified": False, "test_evaluated": False, "provenance": provenance}
    write(results_dir / "summary.json", summary)
    print("Dev", report, flush=True)
    print("Challenge", challenge_report["baseline"], "->", challenge_report["candidate"], flush=True)
    return summary


def activate(results_dir):
    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary["eligible_for_activation"]:
        raise ValueError("Candidate did not pass the frozen promotion gates.")
    if summary.get("deployment_hold"):
        raise ValueError("Candidate is held after diagnostic false-alarm regression; use an isolated preview, not default activation.")
    for p, expected in summary["provenance"]["source_sha256"].items():
        if digest(ROOT / p) != expected:
            raise ValueError("Experiment source changed after validation.")
    if (digest(ROOT / "configs/message_expansion.yaml") != summary["provenance"]["config_sha256"]
            or digest(SPLITS) != summary["provenance"]["uci_split_sha256"]
            or digest(SPLIT) != summary["provenance"]["domain_split_sha256"]):
        raise ValueError("Experiment configuration or split definitions changed.")
    active = ROOT / "models/message_spam.joblib"
    candidate = ROOT / summary["candidate_path"]
    if (hashlib.sha256(active.read_bytes()).hexdigest() != summary["previous_active_sha256"]
            or hashlib.sha256(candidate.read_bytes()).hexdigest() != summary["candidate_sha256"]):
        raise ValueError("Active or candidate model changed after evaluation.")
    load_message_classifier(candidate)
    backup = candidate.parent / "previous_active.joblib"
    for source, target in ((active, backup), (active.with_suffix(".manifest.json"), backup.with_suffix(".manifest.json"))):
        shutil.copy2(source, target)
    # Stage both files before replacement; the module recovers on a paired-file change.
    for suffix in (".joblib", ".manifest.json"):
        shutil.copy2(candidate.with_suffix(suffix), active.with_suffix(suffix + ".pending"))
    for suffix in (".joblib", ".manifest.json"):
        active.with_suffix(suffix + ".pending").replace(active.with_suffix(suffix))
    load_message_classifier(active)
    write(results_dir / "promotion.json", {"previous_sha256": summary["previous_active_sha256"],
          "active_sha256": summary["candidate_sha256"], "backup": str(backup.relative_to(ROOT)), "test_evaluated": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--activate", action="store_true", help="Activate an already evaluated candidate; does not refit.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/message_generalization_v1")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/message_generalization_v1")
    args = parser.parse_args(argv)
    if args.activate:
        activate(args.results_dir); return
    if args.download:
        download_sms(); download_domain()
    if not SPLIT.exists():
        prepare_domain()
    run(args.results_dir, args.model_dir)


if __name__ == "__main__":
    main()
