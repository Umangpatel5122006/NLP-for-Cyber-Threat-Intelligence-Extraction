"""Train-only comparison of background sampling for the English action CRF.

Reuses verified relevant-only OOF action caches. Trains isolated candidates;
the active artifacts and normal training configuration are never replaced.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from cti.core.dataset import load_smah
from cti.core.sms_data import ROOT
from cti.eval.cv import get_report_grouped_splits
from cti.experiment_action import _check_predictions, _digest, _write
from cti.experiment_action_relevance import cache_predictions, score
from cti.models.action_crf import create_crf_tagger
from cti.models.ner_artifact import _environment, load_ner_tagger, save_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner


POLICIES = ("relevant_only", "capped_background", "all_background")


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("version") != 1 or config.get("folds") != 3
            or config.get("seed") != 42 or config.get("candidates") != list(POLICIES)):
        raise ValueError("Use the fixed three-fold, three-policy protocol (seed 42).")
    if not isinstance(config.get("parameters"), dict):
        raise ValueError("Specify the fixed classical CRF parameters.")
    create_crf_tagger(**config["parameters"])
    for key in ("selection", "confirmation"):
        for name in ("minimum_f1_gain", "maximum_exact_recall_loss", "minimum_background_reduction"):
            value = config[key][name]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Selection rates must be finite numbers between zero and one.")
    wins = config["selection"]["minimum_fold_wins"]
    if type(wins) is not int or not 1 <= wins <= 3:
        raise ValueError("Require between one and three fold wins.")
    return config


def training_indices(data, indices, policy, seed=42):
    """Keep all relevant sentences; sample annotated-all-O background per report.

    Capped policy adds at most one background sentence per relevant sentence in
    each report. Selection hashes stable UIDs, independent of input order. No
    held-out report can contribute counts, text, tags or sampled sentences.
    """
    if policy not in POLICIES:
        raise ValueError("Unknown training policy.")
    indices = list(indices)
    if len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(data.tokens) for i in indices):
        raise ValueError("Training indices must be distinct valid integers.")
    positive, background = [], []
    for i in indices:
        if not data.tokens[i]:
            continue
        if data.relevant[i]:
            positive.append(i)
        else:
            if any(tag != "O" for tag in data.tags[i]):
                raise ValueError("Background sentences must have all-O action annotations.")
            background.append(i)
    if not positive:
        raise ValueError("Training requires nonempty relevant sentences.")
    if policy == "relevant_only":
        return sorted(positive)
    if policy == "all_background":
        return sorted(positive + background)
    counts, grouped = {}, {}
    for i in positive:
        counts[data.doc_ids[i]] = counts.get(data.doc_ids[i], 0) + 1
    for i in background:
        grouped.setdefault(data.doc_ids[i], []).append(i)
    selected = list(positive)
    for report, rows in grouped.items():
        ordered = sorted(rows, key=lambda i: (hashlib.sha256(f"{seed}:{data.uids[i]}".encode()).hexdigest(), data.uids[i]))
        selected.extend(ordered[:counts.get(report, 0)])
    return sorted(selected)


def metrics(data, predictions, indices=None):
    rows = list(range(len(data.tokens))) if indices is None else list(indices)
    return score([data.tags[i] for i in rows], [predictions[i] for i in rows],
                 [data.relevant[i] for i in rows], [1.0] * len(rows), None)


def checks(baseline, candidate, policy):
    before, after = baseline["background_predicted_spans"], candidate["background_predicted_spans"]
    reduction = (before - after) / before if before else (0.0 if after == 0 else -1.0)
    return {"f1_gain": candidate["f1"] >= baseline["f1"] + policy["minimum_f1_gain"],
            "exact_recall": candidate["recall"] >= baseline["recall"] - policy["maximum_exact_recall_loss"],
            "background_reduction": reduction >= policy["minimum_background_reduction"]}


def choose(reports, policy):
    baseline, eligible = reports[0], []
    if baseline["id"] != "relevant_only":
        raise ValueError("The reference must be the relevant-only policy.")
    for candidate in reports[1:]:
        candidate["checks"] = checks(baseline["pooled"], candidate["pooled"], policy)
        candidate["fold_wins"] = sum(a["f1"] > b["f1"] for a, b in zip(candidate["folds"], baseline["folds"]))
        candidate["eligible"] = all(candidate["checks"].values()) and candidate["fold_wins"] >= policy["minimum_fold_wins"]
        if candidate["eligible"]:
            eligible.append(candidate)
    # Configured candidate order breaks ties, without inspecting dev.
    return max(eligible, key=lambda row: row["pooled"]["f1"])["id"] if eligible else baseline["id"]


def fit_model(data, indices, policy, parameters, seed):
    selected = training_indices(data, indices, policy, seed)
    model = create_crf_tagger(**parameters)
    model.fit([data.tokens[i] for i in selected], [data.tags[i] for i in selected])
    return model, {"sentences": len(selected), "relevant": sum(data.relevant[i] for i in selected),
                   "background": sum(not data.relevant[i] for i in selected),
                   "uids_sha256": _digest([data.uids[i] for i in selected])}


def run(config_path=ROOT / "configs/action_background_experiment.yaml"):
    config = read_config(config_path)
    output, models = ROOT / "results/action_background_v1", ROOT / "models/action_background_v1"
    output.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    active_path = ROOT / "models/ner_action.joblib"
    active = load_ner_tagger(active_path, expected_layer="action")
    active_files = [active_path, active_path.with_suffix(".manifest.json"), ROOT / "models/ner_entities.joblib",
                    ROOT / "models/message_spam.joblib", ROOT / "models/message_generalization_v1/candidate.joblib",
                    ROOT / "configs/ner_training.yaml", ROOT / "config_ner.yaml"]
    unchanged = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in active_files if p.exists()}
    ids = {s: set(json.loads((ROOT / f"data/splits/{s}_report_ids.json").read_text(encoding="utf-8"))) for s in ("train", "dev", "test")}
    if ids["train"] & (ids["dev"] | ids["test"]) or ids["dev"] & ids["test"]:
        raise ValueError("Official report splits must be disjoint.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    train = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    if set(train.doc_ids) != ids["train"]:
        raise ValueError("Training data must match the official train report IDs.")
    # Validate background annotations before starting any fit.
    training_indices(train, range(len(train.tokens)), "all_background", config["seed"])
    folds = get_report_grouped_splits(train.doc_ids, n_splits=config["folds"], seed=config["seed"])
    reference = json.loads((ROOT / "results/action_features_v1/cv.json").read_text(encoding="utf-8"))
    prior = reference["provenance"]
    if prior["environment"] != _environment() or len(reference["folds"]) != len(folds):
        raise ValueError("Reference action CV environment/fold count differs.")
    for name, expected in prior["source_sha256_lf"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Reference action source changed: " + name)
    if (sha256_file(ROOT / "data/raw/SMAH-CTI.json") != prior["raw_dataset_sha256_lf"]
            or sha256_file(ROOT / "data/processed/bio_en_action_train.json") != prior["processed_train_sha256_lf"]
            or sha256_file(ROOT / "configs/action_features_experiment.yaml") != prior["config_sha256_lf"]):
        raise ValueError("Reference training data/config changed.")
    parameters = next(c["parameters"] for c in reference["candidates"] if c["id"] == "action_boundaries")
    if (parameters != config["parameters"] or any(active.manifest["parameters"].get(k) != v for k, v in parameters.items())
            or active.manifest["metadata"].get("training_policy") != "relevant_only"
            or active.manifest["metadata"]["selection"]["provenance"] != prior
            or set(active.manifest["metadata"]["selection"]["train_report_ids"]) != ids["train"]):
        raise ValueError("Reference/active action parameters or training provenance differ.")
    baseline, fold_records, reference_hashes = [None] * len(train.tokens), [], {}
    for number, ((fit, valid), previous) in enumerate(zip(folds, reference["folds"]), 1):
        record = {"train_report_ids": sorted({train.doc_ids[i] for i in fit}),
                  "validation_report_ids": sorted({train.doc_ids[i] for i in valid}), "validation_sentences": len(valid)}
        if record != previous or set(record["train_report_ids"]) & set(record["validation_report_ids"]):
            raise ValueError("Reference folds must match disjoint train-report folds.")
        path = ROOT / f"models/action_features_v1/action_boundaries_fold_{number}.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        signature = {"fingerprint": _digest(prior), "candidate": {"id": "action_boundaries", "parameters": parameters}, "fold": record}
        predictions = cache_predictions(cache, signature, [train.tokens[i] for i in valid])
        for i, tags in zip(valid, predictions):
            baseline[i] = tags
        fold_records.append(record)
        reference_hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    _check_predictions(train.tokens, baseline)
    base_metrics = metrics(train, baseline)
    original = next(c["pooled"]["exact_strict"] for c in reference["candidates"] if c["id"] == "action_boundaries")
    if any(not math.isclose(base_metrics[k], original[k], abs_tol=0.000051) for k in ("precision", "recall", "f1")):
        raise ValueError("Cached baseline does not reproduce the prior action CV scores.")
    dependencies = ("src/cti/experiment_action_background.py", "src/cti/experiment_action_relevance.py",
                    "src/cti/experiment_action.py", "src/cti/models/action_crf.py", "src/cti/models/crf.py",
                    "src/cti/models/ner_artifact.py", "src/cti/models/ner_data.py", "src/cti/core/dataset.py",
                    "src/cti/eval/cv.py", "src/cti/eval/metrics.py")
    provenance = {"config_sha256_lf": sha256_file(config_path), "environment": _environment(),
                  "source_sha256_lf": {name: sha256_file(ROOT / name) for name in dependencies},
                  "raw_sha256_lf": prior["raw_dataset_sha256_lf"], "processed_train_sha256_lf": prior["processed_train_sha256_lf"],
                  "reference_cache_sha256_lf": reference_hashes, "unchanged_files_sha256": unchanged,
                  "split_sha256_lf": {s: sha256_file(ROOT / f"data/splits/{s}_report_ids.json") for s in ids}}
    fingerprint = _digest(provenance)
    reports = [{"id": "relevant_only", "folds": [metrics(train, baseline, valid) for _, valid in folds], "pooled": base_metrics}]
    _write(models / "relevant_only_oof.json", baseline)
    for policy in POLICIES[1:]:
        predicted, inventories = [None] * len(train.tokens), []
        for number, ((fit, valid), record) in enumerate(zip(folds, fold_records), 1):
            chosen = training_indices(train, fit, policy, config["seed"])
            inventory = {"sentences": len(chosen), "relevant": sum(train.relevant[i] for i in chosen),
                         "background": sum(not train.relevant[i] for i in chosen), "uids_sha256": _digest([train.uids[i] for i in chosen])}
            signature = {"fingerprint": fingerprint, "policy": policy, "fold": record, "training": inventory}
            path = models / f"{policy}_fold_{number}.json"
            cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            tokens = [train.tokens[i] for i in valid]
            if cache.get("signature") == signature:
                decoded = cache_predictions(cache, signature, tokens)
                print(f"{policy}: resumed fold {number}/3", flush=True)
            else:
                print(f"{policy}: fitting fold {number}/3 on {inventory}", flush=True)
                model, fitted_inventory = fit_model(train, fit, policy, parameters, config["seed"])
                if fitted_inventory != inventory:
                    raise ValueError("Sampling changed between planning and fitting.")
                decoded = model.predict(tokens)
                _check_predictions(tokens, decoded)
                _write(path, {"signature": signature, "predictions": decoded, "predictions_sha256": _digest(decoded)})
            for i, tags in zip(valid, decoded):
                predicted[i] = tags
            inventories.append(inventory)
            print(f"  fold exact F1={metrics(train, predicted, valid)['f1']:.6f}", flush=True)
        _check_predictions(train.tokens, predicted)
        report = {"id": policy, "folds": [metrics(train, predicted, valid) for _, valid in folds],
                  "pooled": metrics(train, predicted), "training": inventories}
        reports.append(report)
        _write(models / f"{policy}_oof.json", predicted)
        _write(output / "cv.json", {"provenance": provenance, "folds": fold_records, "candidates": reports})
    winner = choose(reports, config["selection"])
    selection = {"selected": winner, "policy": config["selection"], "candidates": reports, "provenance": provenance,
                 "train_report_ids": sorted(set(train.doc_ids)), "dev_used_for_selection": False, "test_evaluated": False}
    _write(output / "selection.json", selection)
    print("Selection written before dev access:", winner, flush=True)
    summary = {"selected": winner, "dev_evaluated": False, "test_evaluated": False,
               "active_files_modified": False, "provenance": provenance}
    if winner != "relevant_only":
        model, inventory = fit_model(train, range(len(train.tokens)), winner, parameters, config["seed"])
        raw_dev = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["dev"])["dev"]
        dev = load_aligned_ner("action", "dev", raw_dev, ROOT / "data/processed")
        if set(dev.doc_ids) != ids["dev"] or set(dev.uids) & set(train.uids):
            raise ValueError("Dev must cover exactly the disjoint official dev reports.")
        before, after = active.tagger.predict(dev.tokens), model.predict(dev.tokens)
        _check_predictions(dev.tokens, before); _check_predictions(dev.tokens, after)
        baseline_dev, candidate_dev = metrics(dev, before), metrics(dev, after)
        confirmation = checks(baseline_dev, candidate_dev, config["confirmation"])
        path = models / "candidate_action.joblib"
        manifest = save_ner_tagger(model, path, "action", metadata={"selection": selection, "training_policy": winner,
                     "training": inventory, "test_evaluated": False, "processed_dev_sha256_lf": sha256_file(ROOT / "data/processed/bio_en_action_dev.json")})
        if load_ner_tagger(path, expected_layer="action").tagger.predict(dev.tokens) != after:
            raise ValueError("Saved candidate changed dev predictions.")
        dev_report = {"baseline": baseline_dev, "candidate": candidate_dev, "checks": confirmation,
                      "reload_predictions_identical": True, "processed_dev_sha256_lf": sha256_file(ROOT / "data/processed/bio_en_action_dev.json")}
        _write(output / "dev.json", dev_report)
        _write(models / "dev_predictions.json", [{"uid": u, "report": r, "gold": g, "relevant": rel, "baseline": b, "candidate": c}
               for u, r, g, rel, b, c in zip(dev.uids, dev.doc_ids, dev.tags, dev.relevant, before, after)])
        summary.update(dev_evaluated=True, dev=dev_report, quantitative_checks_passed=all(confirmation.values()),
                       candidate_artifact_sha256=manifest["artifact_sha256"], training=inventory,
                       release_status="isolated_candidate_pending_integration_review" if all(confirmation.values()) else "held_failed_dev_confirmation")
        print("Dev confirmation:", dev_report, flush=True)
    else:
        summary["release_status"] = "retain_current_model"
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != value for p, value in unchanged.items()):
        raise ValueError("An active artifact or configuration changed during the experiment.")
    if provenance["source_sha256_lf"] != {p: sha256_file(ROOT / p) for p in dependencies}:
        raise ValueError("Experiment sources changed during fitting.")
    _write(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/action_background_experiment.yaml")
    run(parser.parse_args(argv).config)


if __name__ == "__main__":
    main()
