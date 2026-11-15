"""Train-only action relevance gating comparison; official test never evaluated."""

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import yaml

from cti.core.dataset import load_smah
from cti.core.sms_data import ROOT
from cti.eval.cv import get_report_grouped_splits
from cti.eval.metrics import extract_spans
from cti.experiment_action import _check_predictions, _digest, _write
from cti.models.ner_artifact import _environment, load_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner
from cti.models.relevance import SentenceRelevanceClassifier


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("version") != 1 or config.get("folds") != 3 or config.get("seed") != 42:
        raise ValueError("Use the frozen three-fold action-cache protocol (seed 42).")
    cutoffs = config.get("cutoffs")
    if (not isinstance(cutoffs, list) or not cutoffs or len(set(cutoffs)) != len(cutoffs)
            or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 < t < 1 for t in cutoffs)):
        raise ValueError("Cutoffs must be distinct finite numbers strictly between zero and one.")
    for policy in (config["selection"], config["confirmation"]):
        for name in ("minimum_f1_gain", "maximum_exact_recall_loss", "maximum_relevant_sentence_skip_rate"):
            if type(policy[name]) not in (int, float) or not math.isfinite(policy[name]) or not 0 <= policy[name] <= 1:
                raise ValueError("Gate policy rates must be finite numbers between zero and one.")
    if type(config["selection"]["minimum_fold_wins"]) is not int or not 1 <= config["selection"]["minimum_fold_wins"] <= 3:
        raise ValueError("Specify one to three required training fold wins.")
    SentenceRelevanceClassifier(**config["relevance_parameters"])
    return config


def gate(predictions, scores, cutoff):
    if len(predictions) != len(scores) or any(type(s) not in (float, int) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Action predictions need aligned finite relevance scores.")
    if cutoff is not None and (type(cutoff) not in (int, float) or not math.isfinite(cutoff) or not 0 <= cutoff <= 1):
        raise ValueError("Invalid relevance cutoff.")
    return [list(tags) if cutoff is None or score >= cutoff else ["O"] * len(tags)
            for tags, score in zip(predictions, scores)]


def score(gold, predictions, relevant, scores, cutoff):
    if not (len(gold) == len(predictions) == len(relevant) == len(scores)):
        raise ValueError("Gold, predictions, relevance and scores must align.")
    predicted = gate(predictions, scores, cutoff)
    tp = total_gold = total_pred = background = 0
    for truth, guess, is_relevant in zip(gold, predicted, relevant):
        if len(truth) != len(guess):
            raise ValueError("Gold and predicted token counts differ.")
        a, b = set(extract_spans(truth, mode="strict")), set(extract_spans(guess, mode="strict"))
        tp += len(a & b); total_gold += len(a); total_pred += len(b)
        background += 0 if is_relevant else len(b)
    skipped = sum(cutoff is not None and s < cutoff and r for s, r in zip(scores, relevant))
    return {"tp": tp, "gold_spans": total_gold, "predicted_spans": total_pred,
            "precision": tp / total_pred if total_pred else 0.0,
            "recall": tp / total_gold if total_gold else 0.0,
            "f1": 2 * tp / (total_gold + total_pred) if total_gold + total_pred else 0.0,
            "background_predicted_spans": background,
            "relevant_sentences_skipped": int(skipped), "relevant_sentences": sum(relevant),
            "relevant_sentence_skip_rate": skipped / sum(relevant) if sum(relevant) else 0.0}


def checks(baseline, candidate, policy):
    return {"f1_gain": candidate["f1"] >= baseline["f1"] + policy["minimum_f1_gain"],
            "exact_recall": candidate["recall"] >= baseline["recall"] - policy["maximum_exact_recall_loss"],
            "relevant_sentence_coverage": candidate["relevant_sentence_skip_rate"] <= policy["maximum_relevant_sentence_skip_rate"]}


def choose(reports, policy):
    baseline = reports[0]
    eligible = []
    for report in reports[1:]:
        wins = sum(a["f1"] > b["f1"] for a, b in zip(report["folds"], baseline["folds"]))
        report["checks"] = checks(baseline["pooled"], report["pooled"], policy)
        report["fold_wins"] = wins
        report["eligible"] = all(report["checks"].values()) and wins >= policy["minimum_fold_wins"]
        if report["eligible"]:
            eligible.append(report)
    return max(eligible, key=lambda r: (r["pooled"]["f1"], -r["cutoff"])) if eligible else baseline


def cache_predictions(cache, signature, tokens):
    if cache.get("signature") != signature or cache.get("predictions_sha256") != _digest(cache.get("predictions")):
        raise ValueError("Action OOF cache provenance or predictions changed; regenerate in a separate experiment directory.")
    _check_predictions(tokens, cache["predictions"])
    return cache["predictions"]


def run(config_path=ROOT / "configs/action_relevance_experiment.yaml"):
    config = read_config(config_path)
    output, models = ROOT / "results/action_relevance_v1", ROOT / "models/action_relevance_v1"
    output.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    active_path = ROOT / "models/ner_action.joblib"
    active = load_ner_tagger(active_path, expected_layer="action")
    active_sha = hashlib.sha256(active_path.read_bytes()).hexdigest()
    raw_train = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    train = load_aligned_ner("action", "train", raw_train, ROOT / "data/processed")
    text_by_uid = dict(zip(raw_train["uid"], raw_train["sentence_text"]))
    texts = [text_by_uid[uid] for uid in train.uids]
    folds = get_report_grouped_splits(train.doc_ids, n_splits=3, seed=42)
    reference = json.loads((ROOT / "results/action_features_v1/cv.json").read_text(encoding="utf-8"))
    prior = reference["provenance"]
    if len(reference["folds"]) != len(folds) or len(folds) != config["folds"]:
        raise ValueError("The action cache must cover all three report folds.")
    if prior["environment"] != _environment():
        raise ValueError("Action CV runtime differs from the cached experiment.")
    for name, expected in prior["source_sha256_lf"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Action-cache source changed: " + name)
    if (sha256_file(ROOT / "data/processed/bio_en_action_train.json") != prior["processed_train_sha256_lf"]
            or sha256_file(ROOT / "data/raw/SMAH-CTI.json") != prior["raw_dataset_sha256_lf"]
            or sha256_file(ROOT / "configs/action_features_experiment.yaml") != prior["config_sha256_lf"]):
        raise ValueError("Training corpus differs from the action cache.")
    params = next(r["parameters"] for r in reference["candidates"] if r["id"] == "action_boundaries")
    if (any(active.manifest["parameters"].get(k) != v for k, v in params.items())
            or active.manifest["metadata"].get("training_policy") != "relevant_only"
            or set(active.manifest["metadata"]["selection"]["train_report_ids"]) != set(train.doc_ids)
            or active.manifest["metadata"]["selection"]["provenance"] != prior):
        raise ValueError("The active action model is not the cached train-only action profile.")
    action_predictions, scores = [None] * len(texts), [None] * len(texts)
    cache_hashes = {}
    for number, ((fit, valid), old_fold) in enumerate(zip(folds, reference["folds"]), 1):
        expected_fold = {"train_report_ids": sorted({train.doc_ids[i] for i in fit}),
                         "validation_report_ids": sorted({train.doc_ids[i] for i in valid}), "validation_sentences": len(valid)}
        if expected_fold != old_fold or set(expected_fold["train_report_ids"]) & set(expected_fold["validation_report_ids"]):
            raise ValueError("Action/relevance folds must use the same disjoint training reports.")
        path = ROOT / f"models/action_features_v1/action_boundaries_fold_{number}.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        signature = {"fingerprint": _digest(prior), "candidate": {"id": "action_boundaries", "parameters": params}, "fold": expected_fold}
        predicted = cache_predictions(cache, signature, [train.tokens[i] for i in valid])
        cache_hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        for i, tags in zip(valid, predicted):
            action_predictions[i] = tags
    _check_predictions(train.tokens, action_predictions)
    original_metrics = next(r["pooled"]["exact_strict"] for r in reference["candidates"] if r["id"] == "action_boundaries")
    restored_metrics = score(train.tags, action_predictions, train.relevant, [0.5] * len(train.tags), None)
    if any(not math.isclose(restored_metrics[k], original_metrics[k], abs_tol=0.000051) for k in ("precision", "recall", "f1")):
        raise ValueError("Action cache does not reproduce the frozen training CV metrics.")
    dependencies = ("src/cti/experiment_action_relevance.py", "src/cti/models/relevance.py", "src/cti/models/ner_data.py",
                    "src/cti/eval/cv.py", "src/cti/eval/metrics.py", "src/cti/preprocessing/cleaner.py",
                    "src/cti/preprocessing/defanger.py", "src/cti/core/offsets.py", "src/cti/core/dataset.py")
    provenance = {"config_sha256_lf": sha256_file(config_path), "active_action_sha256": active_sha,
                  "python": platform.python_version(),
                  "packages": {name: version(name) for name in ("scikit-learn", "numpy", "scipy", "joblib", "pandas")},
                  "source_sha256_lf": {p: sha256_file(ROOT / p) for p in dependencies}, "action_cache_sha256_lf": cache_hashes,
                  "processed_train_sha256_lf": prior["processed_train_sha256_lf"], "raw_sha256_lf": prior["raw_dataset_sha256_lf"],
                  "train_report_ids": sorted(set(train.doc_ids)),
                  "split_sha256_lf": {s: sha256_file(ROOT / f"data/splits/{s}_report_ids.json") for s in ("train", "dev", "test")}}
    for number, (fit, valid) in enumerate(folds, 1):
        print(f"Fitting classical relevance fold {number}/3 (training reports only)", flush=True)
        model = SentenceRelevanceClassifier(**config["relevance_parameters"]).fit(
            [texts[i] for i in fit], [int(train.relevant[i]) for i in fit])
        predicted_scores = model.predict_scores([texts[i] for i in valid])
        for i, value in zip(valid, predicted_scores):
            scores[i] = value
    reports = []
    for cutoff in [None, *config["cutoffs"]]:
        by_fold = [score([train.tags[i] for i in valid], [action_predictions[i] for i in valid],
                         [train.relevant[i] for i in valid], [scores[i] for i in valid], cutoff) for _, valid in folds]
        reports.append({"id": "no_gate" if cutoff is None else f"gate_{cutoff}", "cutoff": cutoff,
                        "folds": by_fold, "pooled": score(train.tags, action_predictions, train.relevant, scores, cutoff)})
    selected = choose(reports, config["selection"])
    selection = {"selected": selected["id"], "cutoff": selected["cutoff"], "candidates": reports,
                 "provenance": provenance, "dev_used_for_selection": False, "test_evaluated": False}
    _write(output / "selection.json", selection)
    _write(models / "oof.json", [{"uid": uid, "report": report, "score": value, "relevant": rel, "gold": gold, "action": tags}
                               for uid, report, value, rel, gold, tags in zip(train.uids, train.doc_ids, scores, train.relevant, train.tags, action_predictions)])
    print("Selection written before dev access:", selected["id"], selected["pooled"], flush=True)
    summary = {"selected": selected["id"], "cutoff": selected["cutoff"], "provenance": provenance,
               "dev_evaluated": False, "test_evaluated": False, "active_files_modified": False}
    if selected["cutoff"] is not None:
        full = SentenceRelevanceClassifier(threshold=selected["cutoff"], **config["relevance_parameters"]).fit(
            texts, [int(r) for r in train.relevant])
        raw_dev = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["dev"])["dev"]
        dev = load_aligned_ner("action", "dev", raw_dev, ROOT / "data/processed")
        if set(dev.doc_ids) & set(train.doc_ids) or set(dev.uids) & set(train.uids):
            raise ValueError("Train/dev reports and sentence UIDs must be disjoint.")
        lookup = dict(zip(raw_dev["uid"], raw_dev["sentence_text"]))
        dev_texts = [lookup[uid] for uid in dev.uids]
        dev_scores = full.predict_scores(dev_texts)
        tags = active.tagger.predict(dev.tokens)
        base = score(dev.tags, tags, dev.relevant, dev_scores, None)
        candidate = score(dev.tags, tags, dev.relevant, dev_scores, selected["cutoff"])
        confirmation = checks(base, candidate, config["confirmation"])
        path = models / "relevance.joblib"
        manifest = full.save(path, metadata={"selection": selection, "training_report_ids": sorted(set(train.doc_ids)), "test_evaluated": False})
        if SentenceRelevanceClassifier.load(path).predict_scores(dev_texts) != dev_scores:
            raise ValueError("Reloaded relevance model changed dev scores.")
        summary.update(dev_evaluated=True, baseline_dev=base, candidate_dev=candidate, confirmation_checks=confirmation,
                       quantitative_checks_passed=all(confirmation.values()), relevance_artifact_sha256=manifest["artifact_sha256"])
        _write(output / "dev.json", {"baseline": base, "candidate": candidate, "checks": confirmation, "test_evaluated": False})
        _write(models / "dev_predictions.json", [{"uid": uid, "report": report, "score": value, "relevant": rel, "gold": gold, "action": prediction}
                   for uid, report, value, rel, gold, prediction in zip(dev.uids, dev.doc_ids, dev_scores, dev.relevant, dev.tags, tags)])
        print("Dev confirmation:", summary, flush=True)
    if hashlib.sha256(active_path.read_bytes()).hexdigest() != active_sha or provenance["source_sha256_lf"] != {p: sha256_file(ROOT / p) for p in dependencies}:
        raise ValueError("Action artifact or experiment sources changed during the run.")
    _write(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/action_relevance_experiment.yaml")
    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
