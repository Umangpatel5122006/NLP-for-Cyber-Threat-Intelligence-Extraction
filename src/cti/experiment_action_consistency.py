"""Fit-only ambiguity/deduplication comparison with original validation gold."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import yaml

from cti.core.dataset import load_smah
from cti.core.sms_data import ROOT
from cti.eval.cv import get_report_grouped_splits
from cti.experiment_action import _check_predictions, _digest, _write
from cti.experiment_action_background import metrics
from cti.experiment_action_ending import checks, choose
from cti.experiment_action_relevance import cache_predictions
from cti.models.action_crf import create_crf_tagger
from cti.models.ner_artifact import _environment, load_ner_tagger, save_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner


POLICIES = ("consistent_inputs", "unique_consistent_inputs")


def select_fit_rows(data, indices, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown consistency policy.")
    indices = list(indices)
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(data.tokens) for i in indices):
        raise ValueError("Fit indices must be distinct valid integers.")
    if len(set(data.uids)) != len(data.uids):
        raise ValueError("Sentence UIDs must be unique.")
    groups = defaultdict(list)
    for i in indices:
        if data.relevant[i] and data.tokens[i]:
            groups[tuple(data.tokens[i])].append(i)
    selected, ambiguous, duplicates, ambiguous_groups = [], [], [], 0
    for rows in groups.values():
        if len({tuple(data.tags[i]) for i in rows}) > 1:
            ambiguous.extend(rows)
            ambiguous_groups += 1
        elif policy == "unique_consistent_inputs":
            ordered = sorted(rows, key=lambda i: data.uids[i])
            selected.append(ordered[0])
            duplicates.extend(ordered[1:])
        else:
            selected.extend(rows)
    selected.sort(); ambiguous.sort(); duplicates.sort()
    if not selected:
        raise ValueError("No nonempty relevant fit sentences remain.")
    inventory = {"input_relevant_sentences": sum(map(len, groups.values())), "selected_sentences": len(selected),
                 "ambiguous_groups": ambiguous_groups, "ambiguous_rows_removed": len(ambiguous),
                 "duplicate_rows_removed": len(duplicates), "selected_uids_sha256": _digest([data.uids[i] for i in selected]),
                 "ambiguous_uids_sha256": _digest([data.uids[i] for i in ambiguous]),
                 "duplicate_uids_sha256": _digest([data.uids[i] for i in duplicates])}
    return selected, inventory


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("version") != 1 or config.get("folds") != 3
            or config.get("seed") != 42 or config.get("candidates") != list(POLICIES)):
        raise ValueError("Use the fixed two-policy, three-fold consistency protocol.")
    create_crf_tagger(**config["parameters"])
    for section in ("selection", "confirmation"):
        for name in ("minimum_f1_gain", "maximum_exact_recall_loss", "maximum_background_increase"):
            value = config[section][name]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Selection budgets must be finite rates between zero and one.")
    wins = config["selection"]["minimum_fold_wins"]
    if type(wins) is not int or not 1 <= wins <= 3:
        raise ValueError("Specify one to three required fold wins.")
    return config


def fit(data, indices, policy, parameters):
    selected, inventory = select_fit_rows(data, indices, policy)
    model = create_crf_tagger(**parameters)
    model.fit([data.tokens[i] for i in selected], [data.tags[i] for i in selected])
    return model, inventory


def run(config_path=ROOT / "configs/action_consistency_experiment.yaml"):
    config = read_config(config_path)
    output, models = ROOT / "results/action_consistency_v1", ROOT / "models/action_consistency_v1"
    output.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    upstream = ROOT / "results/action_background_v1"
    original, audited, reference = read(upstream / "summary.json"), read(upstream / "integrity_audit.json"), read(upstream / "selection.json")
    prior = original["provenance"]
    if (not audited["passed"] or original["selected"] != "relevant_only" or reference["provenance"] != prior
            or prior["environment"] != _environment()
            or audited["audit_script_sha256_lf"] != sha256_file(ROOT / "scripts/audit_action_background.py")):
        raise ValueError("The audited reference runtime/provenance differs.")
    for key in ("source_sha256_lf", "reference_cache_sha256_lf"):
        for name, expected in prior[key].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError("Reference source/cache changed: " + name)
    unchanged = dict(prior["unchanged_files_sha256"])
    unchanged["configs/config_ner.yaml"] = hashlib.sha256((ROOT / "configs/config_ner.yaml").read_bytes()).hexdigest()
    for name, expected in unchanged.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Active file changed: " + name)
    for name, key in (("data/raw/SMAH-CTI.json", "raw_sha256_lf"), ("data/processed/bio_en_action_train.json", "processed_train_sha256_lf")):
        if sha256_file(ROOT / name) != prior[key]:
            raise ValueError("Original training data changed.")
    for split, expected in prior["split_sha256_lf"].items():
        if sha256_file(ROOT / f"data/splits/{split}_report_ids.json") != expected:
            raise ValueError("Official report splits changed.")
    previous_config = ROOT / "configs/action_background_experiment.yaml"
    if (sha256_file(previous_config) != prior["config_sha256_lf"]
            or yaml.safe_load(previous_config.read_text(encoding="utf-8"))["parameters"] != config["parameters"]):
        raise ValueError("Keep the reference features/hyperparameters fixed.")
    active = load_ner_tagger(ROOT / "models/ner_action.joblib", expected_layer="action")
    if (active.manifest["metadata"].get("training_policy") != "relevant_only"
            or any(active.manifest["parameters"].get(k) != v for k, v in config["parameters"].items())):
        raise ValueError("The active model must match the original reference profile.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    train = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    ids = {s: set(read(ROOT / f"data/splits/{s}_report_ids.json")) for s in ("train", "dev", "test")}
    if (set(train.doc_ids) != ids["train"] or ids["train"] != set(reference["train_report_ids"])
            or ids["train"] & (ids["dev"] | ids["test"]) or ids["dev"] & ids["test"]):
        raise ValueError("Official train reports must match the disjoint reference.")
    baseline_path = ROOT / "models/action_background_v1/relevant_only_oof.json"
    if sha256_file(baseline_path) != audited["oof_sha256_lf"][baseline_path.relative_to(ROOT).as_posix()]:
        raise ValueError("Audited baseline predictions changed.")
    baseline = read(baseline_path)
    _check_predictions(train.tokens, baseline)
    folds = get_report_grouped_splits(train.doc_ids, n_splits=3, seed=42)
    records = [{"train_report_ids": sorted({train.doc_ids[i] for i in a}), "validation_report_ids": sorted({train.doc_ids[i] for i in b}),
                "validation_sentences": len(b)} for a, b in folds]
    base = next(row for row in reference["candidates"] if row["id"] == "relevant_only")
    if (records != read(upstream / "cv.json")["folds"] or metrics(train, baseline) != base["pooled"]
            or [metrics(train, baseline, b) for _, b in folds] != base["folds"]):
        raise ValueError("Baseline folds/counts must reproduce frozen CV.")
    sources = ("src/cti/experiment_action_consistency.py", "src/cti/experiment_action.py", "src/cti/experiment_action_background.py",
               "src/cti/experiment_action_ending.py", "src/cti/experiment_action_relevance.py", "src/cti/models/ner_data.py",
               "src/cti/models/ner_artifact.py", "src/cti/models/action_crf.py", "src/cti/eval/metrics.py", "src/cti/eval/cv.py", "src/cti/core/dataset.py")
    provenance = {"config_sha256_lf": sha256_file(config_path), "source_sha256_lf": {name: sha256_file(ROOT / name) for name in sources},
                  "environment": _environment(), "raw_sha256_lf": prior["raw_sha256_lf"], "processed_train_sha256_lf": prior["processed_train_sha256_lf"],
                  "split_sha256_lf": prior["split_sha256_lf"], "unchanged_files_sha256": unchanged,
                  "baseline_oof_sha256_lf": sha256_file(baseline_path), "upstream_summary_sha256_lf": sha256_file(upstream / "summary.json"),
                  "upstream_audit_sha256_lf": sha256_file(upstream / "integrity_audit.json")}
    fingerprint = _digest(provenance)
    reports = [{"id": "baseline", "pooled": metrics(train, baseline), "folds": [metrics(train, baseline, b) for _, b in folds]}]
    _write(models / "baseline_oof.json", baseline)
    for policy in POLICIES:
        predictions, inventories = [None] * len(train.tokens), []
        for number, ((fit_rows, valid), record) in enumerate(zip(folds, records), 1):
            _, inventory = select_fit_rows(train, fit_rows, policy)
            signature = {"fingerprint": fingerprint, "policy": policy, "fold": record, "training": inventory}
            path, tokens = models / f"{policy}_fold_{number}.json", [train.tokens[i] for i in valid]
            cached = read(path) if path.exists() else {}
            if cached.get("signature") == signature:
                decoded = cache_predictions(cached, signature, tokens)
                print(f"{policy}: resumed fold {number}/3", flush=True)
            else:
                print(f"{policy}: fitting fold {number}/3; inventory={inventory}", flush=True)
                model, fitted_inventory = fit(train, fit_rows, policy, config["parameters"])
                if fitted_inventory != inventory:
                    raise ValueError("Fit selection changed during the run.")
                decoded = model.predict(tokens)
                _check_predictions(tokens, decoded)
                _write(path, {"signature": signature, "predictions": decoded, "predictions_sha256": _digest(decoded)})
            for i, tags in zip(valid, decoded):
                predictions[i] = tags
            inventories.append(inventory)
            print("  exact F1=", metrics(train, predictions, valid)["f1"], flush=True)
        _check_predictions(train.tokens, predictions)
        reports.append({"id": policy, "pooled": metrics(train, predictions), "folds": [metrics(train, predictions, b) for _, b in folds], "training": inventories})
        _write(models / f"{policy}_oof.json", predictions)
        _write(output / "cv.json", {"provenance": provenance, "folds": records, "candidates": reports})
    winner = choose(reports, config["selection"])
    selection = {"selected": winner, "policy": config["selection"], "provenance": provenance, "candidates": reports,
                 "train_report_ids": sorted(set(train.doc_ids)), "validation_filtered": False, "gold_labels_modified": False,
                 "dev_used_for_selection": False, "test_evaluated": False}
    _write(output / "selection.json", selection)
    print("Selection frozen before dev access:", winner, flush=True)
    summary = {"selected": winner, "provenance": provenance, "dev_evaluated": False, "test_evaluated": False,
               "active_files_modified": False, "gold_labels_modified": False, "validation_filtered": False}
    if winner != "baseline":
        model, inventory = fit(train, range(len(train.tokens)), winner, config["parameters"])
        raw_dev = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["dev"])["dev"]
        dev = load_aligned_ner("action", "dev", raw_dev, ROOT / "data/processed")
        if set(dev.doc_ids) != ids["dev"] or set(dev.uids) & set(train.uids):
            raise ValueError("Official dev must be disjoint and complete.")
        a, b = active.tagger.predict(dev.tokens), model.predict(dev.tokens)
        _check_predictions(dev.tokens, a); _check_predictions(dev.tokens, b)
        before, after = metrics(dev, a), metrics(dev, b)
        confirmation = checks(before, after, config["confirmation"])
        path = models / "candidate_action.joblib"
        manifest = save_ner_tagger(model, path, "action", metadata={"selection": selection, "training_policy": winner, "training": inventory, "test_evaluated": False})
        if load_ner_tagger(path, expected_layer="action").tagger.predict(dev.tokens) != b:
            raise ValueError("Saved candidate changed dev predictions.")
        report = {"baseline": before, "candidate": after, "checks": confirmation, "reload_predictions_identical": True,
                  "processed_dev_sha256_lf": sha256_file(ROOT / "data/processed/bio_en_action_dev.json")}
        _write(output / "dev.json", report)
        _write(models / "dev_predictions.json", [{"uid": uid, "report": doc, "gold": gold, "relevant": rel, "baseline": old, "candidate": new}
               for uid, doc, gold, rel, old, new in zip(dev.uids, dev.doc_ids, dev.tags, dev.relevant, a, b)])
        summary.update(dev_evaluated=True, dev=report, training=inventory, candidate_artifact_sha256=manifest["artifact_sha256"], quantitative_checks_passed=all(confirmation.values()),
                       release_status="isolated_candidate_pending_integration_review" if all(confirmation.values()) else "held_failed_dev_confirmation")
        print("Dev confirmation:", report, flush=True)
    else:
        summary["release_status"] = "retain_current_model"
    for name, expected in unchanged.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Active artifacts/configuration changed during fitting.")
    if provenance["source_sha256_lf"] != {name: sha256_file(ROOT / name) for name in sources}:
        raise ValueError("Experiment source changed during fitting.")
    _write(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/action_consistency_experiment.yaml")
    run(parser.parse_args(argv).config)


if __name__ == "__main__":
    main()
