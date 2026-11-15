"""Fixed action-ending CRF experiment: train-CV selection, conditional dev check."""

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
from cti.experiment_action_background import metrics
from cti.experiment_action_relevance import cache_predictions
from cti.models.action_end_crf import ActionEndCRFTagger, load_candidate, save_candidate
from cti.models.ner_artifact import _environment, load_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = [{"id": "ending_cues", "learned_edges": False}, {"id": "ending_edges", "learned_edges": True}]
    if (not isinstance(config, dict) or config.get("version") != 1 or config.get("folds") != 3
            or config.get("seed") != 42 or config.get("candidates") != expected
            or any(type(row["learned_edges"]) is not bool for row in config["candidates"])):
        raise ValueError("Use the fixed two-profile, three-fold ending experiment.")
    ActionEndCRFTagger(**config["parameters"])
    for section in ("selection", "confirmation"):
        for name in ("minimum_f1_gain", "maximum_exact_recall_loss", "maximum_background_increase"):
            value = config[section][name]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Selection rates must be finite numbers between zero and one.")
    wins = config["selection"]["minimum_fold_wins"]
    if type(wins) is not int or not 1 <= wins <= 3:
        raise ValueError("Specify between one and three fold wins.")
    return config


def checks(base, candidate, policy):
    return {"f1_gain": candidate["f1"] >= base["f1"] + policy["minimum_f1_gain"],
            "exact_recall": candidate["recall"] >= base["recall"] - policy["maximum_exact_recall_loss"],
            "background_budget": candidate["background_predicted_spans"] <= base["background_predicted_spans"] * (1 + policy["maximum_background_increase"])}


def choose(reports, policy):
    base, eligible = reports[0], []
    if base["id"] != "baseline":
        raise ValueError("The current model's OOF reference must come first.")
    for report in reports[1:]:
        report["checks"] = checks(base["pooled"], report["pooled"], policy)
        report["fold_wins"] = sum(a["f1"] > b["f1"] for a, b in zip(report["folds"], base["folds"]))
        report["eligible"] = all(report["checks"].values()) and report["fold_wins"] >= policy["minimum_fold_wins"]
        if report["eligible"]:
            eligible.append(report)
    return max(eligible, key=lambda r: r["pooled"]["f1"])["id"] if eligible else "baseline"


def fit(candidate, data, indices, parameters):
    selected = [i for i in indices if data.tokens[i] and data.relevant[i]]
    if not selected:
        raise ValueError("A fit fold needs relevant nonempty training sentences.")
    model = ActionEndCRFTagger(learned_edges=candidate["learned_edges"], **parameters)
    model.fit([data.tokens[i] for i in selected], [data.tags[i] for i in selected])
    return model


def run(config_path=ROOT / "configs/action_ending_experiment.yaml"):
    config = read_config(config_path)
    output, models = ROOT / "results/action_ending_v1", ROOT / "models/action_ending_v1"
    output.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    upstream = ROOT / "results/action_background_v1"
    frozen, audited, selection = read(upstream / "summary.json"), read(upstream / "integrity_audit.json"), read(upstream / "selection.json")
    prior = frozen["provenance"]
    if (not audited["passed"] or frozen["selected"] != "relevant_only" or selection["provenance"] != prior
            or prior["environment"] != _environment()
            or audited["audit_script_sha256_lf"] != sha256_file(ROOT / "scripts/audit_action_background.py")):
        raise ValueError("The background experiment must have a compatible audited current-model baseline.")
    for group in ("source_sha256_lf", "reference_cache_sha256_lf"):
        for name, expected in prior[group].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError("Upstream source/cache changed: " + name)
    for name, expected in prior["unchanged_files_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Active artifact/config changed: " + name)
    for split, expected in prior["split_sha256_lf"].items():
        if sha256_file(ROOT / f"data/splits/{split}_report_ids.json") != expected:
            raise ValueError("Official split definitions changed.")
    for name, key in (("data/raw/SMAH-CTI.json", "raw_sha256_lf"), ("data/processed/bio_en_action_train.json", "processed_train_sha256_lf")):
        if sha256_file(ROOT / name) != prior[key]:
            raise ValueError("Training corpus changed.")
    previous_config = ROOT / "configs/action_background_experiment.yaml"
    if (sha256_file(previous_config) != prior["config_sha256_lf"]
            or yaml.safe_load(previous_config.read_text(encoding="utf-8"))["parameters"] != config["parameters"]):
        raise ValueError("Keep the frozen reference CRF hyperparameters/features.")
    active = load_ner_tagger(ROOT / "models/ner_action.joblib", expected_layer="action")
    if active.manifest["metadata"].get("training_policy") != "relevant_only" or any(active.manifest["parameters"].get(k) != v for k, v in config["parameters"].items()):
        raise ValueError("The reference must match the active relevant-only CRF.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    train = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    if set(train.doc_ids) != set(selection["train_report_ids"]):
        raise ValueError("Official train reports differ from the audited reference.")
    ids = {s: set(read(ROOT / f"data/splits/{s}_report_ids.json")) for s in ("train", "dev", "test")}
    if set(train.doc_ids) != ids["train"] or ids["train"] & (ids["dev"] | ids["test"]) or ids["dev"] & ids["test"]:
        raise ValueError("Official report splits must be disjoint and match train.")
    baseline_path = ROOT / "models/action_background_v1/relevant_only_oof.json"
    if sha256_file(baseline_path) != audited["oof_sha256_lf"][baseline_path.relative_to(ROOT).as_posix()]:
        raise ValueError("Audited baseline OOF predictions changed.")
    baseline = read(baseline_path)
    _check_predictions(train.tokens, baseline)
    folds = get_report_grouped_splits(train.doc_ids, n_splits=config["folds"], seed=config["seed"])
    old_cv = read(upstream / "cv.json")
    fold_records = [{"train_report_ids": sorted({train.doc_ids[i] for i in fit_rows}),
                     "validation_report_ids": sorted({train.doc_ids[i] for i in valid}), "validation_sentences": len(valid)} for fit_rows, valid in folds]
    if fold_records != old_cv["folds"]:
        raise ValueError("New and reference report folds must match.")
    reference = next(row for row in selection["candidates"] if row["id"] == "relevant_only")
    if metrics(train, baseline) != reference["pooled"] or [metrics(train, baseline, valid) for _, valid in folds] != reference["folds"]:
        raise ValueError("Reference strict counts do not reproduce frozen training CV.")
    dependencies = ("src/cti/experiment_action_ending.py", "src/cti/models/action_end_crf.py",
                    "src/cti/models/action_crf.py", "src/cti/models/ner_artifact.py", "src/cti/models/ner_data.py",
                    "src/cti/eval/cv.py", "src/cti/eval/metrics.py", "src/cti/core/dataset.py",
                    "src/cti/experiment_action_background.py", "src/cti/experiment_action_relevance.py", "src/cti/experiment_action.py")
    provenance = {"config_sha256_lf": sha256_file(config_path), "environment": _environment(),
                  "source_sha256_lf": {name: sha256_file(ROOT / name) for name in dependencies},
                  "upstream_summary_sha256_lf": sha256_file(upstream / "summary.json"), "upstream_audit_sha256_lf": sha256_file(upstream / "integrity_audit.json"),
                  "baseline_oof_sha256_lf": sha256_file(baseline_path), "unchanged_files_sha256": prior["unchanged_files_sha256"],
                  "raw_sha256_lf": prior["raw_sha256_lf"], "processed_train_sha256_lf": prior["processed_train_sha256_lf"],
                  "split_sha256_lf": prior["split_sha256_lf"]}
    fingerprint = _digest(provenance)
    reports = [{"id": "baseline", "pooled": metrics(train, baseline), "folds": [metrics(train, baseline, valid) for _, valid in folds]}]
    _write(models / "baseline_oof.json", baseline)
    for candidate in config["candidates"]:
        predictions = [None] * len(train.tokens)
        for number, ((fit_rows, valid), record) in enumerate(zip(folds, fold_records), 1):
            fitted = [i for i in fit_rows if train.tokens[i] and train.relevant[i]]
            training = {"relevant_sentences": len(fitted), "uids_sha256": _digest([train.uids[i] for i in fitted])}
            signature = {"fingerprint": fingerprint, "candidate": candidate, "fold": record, "training": training}
            path = models / f"{candidate['id']}_fold_{number}.json"
            cached = read(path) if path.exists() else {}
            tokens = [train.tokens[i] for i in valid]
            if cached.get("signature") == signature:
                decoded = cache_predictions(cached, signature, tokens)
                print(f"{candidate['id']}: resumed fold {number}/3", flush=True)
            else:
                print(f"{candidate['id']}: fitting fold {number}/3 on {len(fitted)} relevant sentences", flush=True)
                model = fit(candidate, train, fit_rows, config["parameters"])
                decoded = model.predict(tokens)
                _check_predictions(tokens, decoded)
                _write(path, {"signature": signature, "predictions": decoded, "predictions_sha256": _digest(decoded)})
            for i, tags in zip(valid, decoded):
                predictions[i] = tags
            print("  exact F1=", metrics(train, predictions, valid)["f1"], flush=True)
        _check_predictions(train.tokens, predictions)
        reports.append({"id": candidate["id"], "pooled": metrics(train, predictions),
                        "folds": [metrics(train, predictions, valid) for _, valid in folds]})
        _write(models / f"{candidate['id']}_oof.json", predictions)
        _write(output / "cv.json", {"provenance": provenance, "folds": fold_records, "candidates": reports})
    winner = choose(reports, config["selection"])
    selected = {"selected": winner, "policy": config["selection"], "provenance": provenance,
                "candidates": reports, "train_report_ids": sorted(set(train.doc_ids)), "dev_used_for_selection": False, "test_evaluated": False}
    _write(output / "selection.json", selected)
    print("Selection frozen before dev access:", winner, flush=True)
    summary = {"selected": winner, "provenance": provenance, "dev_evaluated": False, "test_evaluated": False, "active_files_modified": False}
    if winner != "baseline":
        candidate = next(row for row in config["candidates"] if row["id"] == winner)
        model = fit(candidate, train, range(len(train.tokens)), config["parameters"])
        raw_dev = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["dev"])["dev"]
        dev = load_aligned_ner("action", "dev", raw_dev, ROOT / "data/processed")
        if set(dev.doc_ids) != ids["dev"] or set(dev.uids) & set(train.uids):
            raise ValueError("Dev must match disjoint official reports.")
        before, after = active.tagger.predict(dev.tokens), model.predict(dev.tokens)
        _check_predictions(dev.tokens, before); _check_predictions(dev.tokens, after)
        a, b = metrics(dev, before), metrics(dev, after)
        confirmation = checks(a, b, config["confirmation"])
        path = models / "candidate_action.joblib"
        manifest = save_candidate(model, path, {"selection": selected, "training_policy": "relevant_only", "test_evaluated": False})
        if load_candidate(path).tagger.predict(dev.tokens) != after:
            raise ValueError("Saved action-ending candidate changed dev predictions.")
        report = {"baseline": a, "candidate": b, "checks": confirmation, "reload_predictions_identical": True,
                  "processed_dev_sha256_lf": sha256_file(ROOT / "data/processed/bio_en_action_dev.json")}
        _write(output / "dev.json", report)
        _write(models / "dev_predictions.json", [{"uid": uid, "report": doc, "gold": gold, "relevant": rel, "baseline": old, "candidate": new}
               for uid, doc, gold, rel, old, new in zip(dev.uids, dev.doc_ids, dev.tags, dev.relevant, before, after)])
        summary.update(dev_evaluated=True, dev=report, quantitative_checks_passed=all(confirmation.values()), candidate_artifact_sha256=manifest["artifact_sha256"],
                       release_status="isolated_candidate_pending_integration_review" if all(confirmation.values()) else "held_failed_dev_confirmation")
        print("Dev confirmation:", report, flush=True)
    else:
        summary["release_status"] = "retain_current_model"
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != value for p, value in provenance["unchanged_files_sha256"].items()):
        raise ValueError("Active models/configuration changed during experiment.")
    if provenance["source_sha256_lf"] != {p: sha256_file(ROOT / p) for p in dependencies}:
        raise ValueError("Experiment sources changed during fitting.")
    _write(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/action_ending_experiment.yaml")
    run(parser.parse_args(argv).config)


if __name__ == "__main__":
    main()
