"""Single fixed BILOU-vs-BIO CRF comparison, with train-only selection."""

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
from cti.experiment_action_ending import checks, choose
from cti.models.action_bilou import bilou_is_valid, bilou_to_bio, bio_to_bilou
from cti.models.action_bilou_crf import ActionBILOUCRFTagger, load_candidate, save_candidate
from cti.models.ner_artifact import _environment, load_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or (config.get("version"), config.get("folds"), config.get("seed"), config.get("candidate")) != (1, 3, 42, "bilou"):
        raise ValueError("Use the frozen single-candidate BILOU protocol.")
    ActionBILOUCRFTagger(**config["parameters"])
    for section in ("selection", "confirmation"):
        for name in ("minimum_f1_gain", "maximum_exact_recall_loss", "maximum_background_increase"):
            value = config[section][name]
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Selection budgets must be finite rates between zero and one.")
    wins = config["selection"]["minimum_fold_wins"]
    if type(wins) is not int or not 1 <= wins <= 3:
        raise ValueError("Require one to three fold wins.")
    return config


def decode_cache(cache, signature, tokens):
    internal, predicted = cache.get("internal"), cache.get("predictions")
    if (cache.get("signature") != signature or cache.get("internal_sha256") != _digest(internal)
            or cache.get("predictions_sha256") != _digest(predicted) or not isinstance(internal, list)
            or len(internal) != len(tokens) or any(not isinstance(row, list) or len(row) != len(words) for row, words in zip(internal, tokens))):
        raise ValueError("BILOU cache provenance, hashes or alignment changed.")
    if [bilou_to_bio(row) for row in internal] != predicted:
        raise ValueError("BILOU cache does not reproduce the frozen BIO normalization.")
    _check_predictions(tokens, predicted)
    return internal, predicted


def fit(data, rows, parameters):
    selected = [i for i in rows if data.tokens[i] and data.relevant[i]]
    if not selected:
        raise ValueError("Fit requires nonempty relevant training sentences.")
    return ActionBILOUCRFTagger(**parameters).fit([data.tokens[i] for i in selected], [data.tags[i] for i in selected])


def run(config_path=ROOT / "configs/action_bilou_experiment.yaml"):
    config = read_config(config_path)
    output, models = ROOT / "results/action_bilou_v1", ROOT / "models/action_bilou_v1"
    output.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    upstream = ROOT / "results/action_background_v1"
    frozen, audited, selected = read(upstream / "summary.json"), read(upstream / "integrity_audit.json"), read(upstream / "selection.json")
    prior = frozen["provenance"]
    if (not audited["passed"] or frozen["selected"] != "relevant_only" or selected["provenance"] != prior
            or prior["environment"] != _environment()
            or audited["audit_script_sha256_lf"] != sha256_file(ROOT / "scripts/audit_action_background.py")):
        raise ValueError("Use the compatible audited current-model reference.")
    for group in ("source_sha256_lf", "reference_cache_sha256_lf"):
        for name, expected in prior[group].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError("Reference source/cache changed: " + name)
    unchanged = dict(prior["unchanged_files_sha256"])
    # Include the actual document-pipeline config as well as normal training.
    unchanged["configs/config_ner.yaml"] = hashlib.sha256((ROOT / "configs/config_ner.yaml").read_bytes()).hexdigest()
    for name, expected in unchanged.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Active artifact/config changed: " + name)
    for name, key in (("data/raw/SMAH-CTI.json", "raw_sha256_lf"), ("data/processed/bio_en_action_train.json", "processed_train_sha256_lf")):
        if sha256_file(ROOT / name) != prior[key]:
            raise ValueError("Training corpus differs from reference.")
    for split, expected in prior["split_sha256_lf"].items():
        if sha256_file(ROOT / f"data/splits/{split}_report_ids.json") != expected:
            raise ValueError("Official splits changed.")
    old_config = ROOT / "configs/action_background_experiment.yaml"
    if (sha256_file(old_config) != prior["config_sha256_lf"]
            or yaml.safe_load(old_config.read_text(encoding="utf-8"))["parameters"] != config["parameters"]):
        raise ValueError("Keep the reference features and CRF hyperparameters fixed.")
    active = load_ner_tagger(ROOT / "models/ner_action.joblib", expected_layer="action")
    if (active.manifest["metadata"].get("training_policy") != "relevant_only"
            or any(active.manifest["parameters"].get(k) != v for k, v in config["parameters"].items())):
        raise ValueError("Current model must match the relevant-only reference profile.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    train = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    ids = {s: set(read(ROOT / f"data/splits/{s}_report_ids.json")) for s in ("train", "dev", "test")}
    if (set(train.doc_ids) != ids["train"] or ids["train"] != set(selected["train_report_ids"])
            or ids["train"] & (ids["dev"] | ids["test"]) or ids["dev"] & ids["test"]):
        raise ValueError("Official report splits must be disjoint and match reference.")
    for tags in train.tags:
        if bilou_to_bio(bio_to_bilou(tags)) != tags:
            raise ValueError("Encoding must preserve every gold BIO token and span.")
    reference_path = ROOT / "models/action_background_v1/relevant_only_oof.json"
    if sha256_file(reference_path) != audited["oof_sha256_lf"][reference_path.relative_to(ROOT).as_posix()]:
        raise ValueError("Audited reference OOF changed.")
    baseline = read(reference_path)
    _check_predictions(train.tokens, baseline)
    folds = get_report_grouped_splits(train.doc_ids, n_splits=config["folds"], seed=config["seed"])
    records = [{"train_report_ids": sorted({train.doc_ids[i] for i in a}), "validation_report_ids": sorted({train.doc_ids[i] for i in b}),
                "validation_sentences": len(b)} for a, b in folds]
    reference = next(row for row in selected["candidates"] if row["id"] == "relevant_only")
    if (records != read(upstream / "cv.json")["folds"] or metrics(train, baseline) != reference["pooled"]
            or [metrics(train, baseline, b) for _, b in folds] != reference["folds"]):
        raise ValueError("Reference folds/counts must reproduce frozen training CV.")
    sources = ("src/cti/experiment_action_bilou.py", "src/cti/models/action_bilou_crf.py", "src/cti/models/action_bilou.py", "src/cti/models/action_crf.py",
               "src/cti/models/ner_artifact.py", "src/cti/models/ner_data.py", "src/cti/core/dataset.py", "src/cti/eval/cv.py",
               "src/cti/eval/metrics.py", "src/cti/experiment_action.py", "src/cti/experiment_action_background.py",
               "src/cti/experiment_action_relevance.py", "src/cti/experiment_action_ending.py")
    provenance = {"config_sha256_lf": sha256_file(config_path), "environment": _environment(),
                  "source_sha256_lf": {p: sha256_file(ROOT / p) for p in sources}, "unchanged_files_sha256": unchanged,
                  "raw_sha256_lf": prior["raw_sha256_lf"], "processed_train_sha256_lf": prior["processed_train_sha256_lf"],
                  "split_sha256_lf": prior["split_sha256_lf"], "baseline_oof_sha256_lf": sha256_file(reference_path),
                  "upstream_summary_sha256_lf": sha256_file(upstream / "summary.json"), "upstream_audit_sha256_lf": sha256_file(upstream / "integrity_audit.json")}
    fingerprint, predictions, all_internal = _digest(provenance), [None] * len(train.tokens), [None] * len(train.tokens)
    for number, ((fit_rows, valid), record) in enumerate(zip(folds, records), 1):
        fitted = [i for i in fit_rows if train.tokens[i] and train.relevant[i]]
        signature = {"fingerprint": fingerprint, "fold": record, "candidate": "bilou", "training": {
                     "relevant_sentences": len(fitted), "uids_sha256": _digest([train.uids[i] for i in fitted])}}
        path, tokens = models / f"bilou_fold_{number}.json", [train.tokens[i] for i in valid]
        if path.exists() and read(path).get("signature") == signature:
            internal, decoded = decode_cache(read(path), signature, tokens)
            print(f"BILOU: resumed fold {number}/3", flush=True)
        else:
            print(f"BILOU: fitting fold {number}/3 on {len(fitted)} relevant sentences", flush=True)
            model = fit(train, fit_rows, config["parameters"])
            internal = model.predict_internal(tokens)
            decoded = [bilou_to_bio(row) for row in internal]
            _check_predictions(tokens, decoded)
            _write(path, {"signature": signature, "internal": internal, "internal_sha256": _digest(internal),
                          "predictions": decoded, "predictions_sha256": _digest(decoded)})
        for i, raw_tags, tags in zip(valid, internal, decoded):
            all_internal[i], predictions[i] = raw_tags, tags
        print("  exact F1=", metrics(train, predictions, valid)["f1"], flush=True)
    _check_predictions(train.tokens, predictions)
    reports = [{"id": "baseline", "pooled": metrics(train, baseline), "folds": [metrics(train, baseline, b) for _, b in folds]},
               {"id": "bilou", "pooled": metrics(train, predictions), "folds": [metrics(train, predictions, b) for _, b in folds],
                "internal_invalid_sentences": sum(not bilou_is_valid(row) for row in all_internal)}]
    _write(models / "baseline_oof.json", baseline)
    _write(models / "bilou_oof.json", predictions)
    _write(models / "internal_oof.json", all_internal)
    winner = choose(reports, config["selection"])
    selection = {"selected": winner, "policy": config["selection"], "candidates": reports, "provenance": provenance,
                 "train_report_ids": sorted(set(train.doc_ids)), "gold_roundtrip_identical": True, "dev_used_for_selection": False, "test_evaluated": False}
    _write(output / "cv.json", {"provenance": provenance, "folds": records, "candidates": reports})
    _write(output / "selection.json", selection)
    print("Selection frozen before dev access:", winner, flush=True)
    summary = {"selected": winner, "provenance": provenance, "dev_evaluated": False, "test_evaluated": False,
               "active_files_modified": False, "gold_roundtrip_identical": True}
    if winner != "baseline":
        model = fit(train, range(len(train.tokens)), config["parameters"])
        raw_dev = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["dev"])["dev"]
        dev = load_aligned_ner("action", "dev", raw_dev, ROOT / "data/processed")
        if set(dev.doc_ids) != ids["dev"] or set(dev.uids) & set(train.uids):
            raise ValueError("Official dev must be disjoint from train.")
        before, internal = active.tagger.predict(dev.tokens), model.predict_internal(dev.tokens)
        after = [bilou_to_bio(row) for row in internal]
        _check_predictions(dev.tokens, before); _check_predictions(dev.tokens, after)
        a, b = metrics(dev, before), metrics(dev, after)
        confirmation = checks(a, b, config["confirmation"])
        path = models / "candidate_action.joblib"
        manifest = save_candidate(model, path, {"selection": selection, "training_policy": "relevant_only", "test_evaluated": False})
        restored, _ = load_candidate(path)
        if restored.predict(dev.tokens) != after:
            raise ValueError("Reloaded BILOU candidate changed BIO predictions.")
        report = {"baseline": a, "candidate": b, "checks": confirmation, "reload_predictions_identical": True,
                  "internal_invalid_sentences": sum(not bilou_is_valid(row) for row in internal),
                  "processed_dev_sha256_lf": sha256_file(ROOT / "data/processed/bio_en_action_dev.json")}
        _write(output / "dev.json", report)
        _write(models / "dev_predictions.json", [{"uid": uid, "report": doc, "gold": gold, "relevant": rel, "baseline": old, "internal": raw_tags, "candidate": new}
               for uid, doc, gold, rel, old, raw_tags, new in zip(dev.uids, dev.doc_ids, dev.tags, dev.relevant, before, internal, after)])
        summary.update(dev_evaluated=True, dev=report, quantitative_checks_passed=all(confirmation.values()), candidate_artifact_sha256=manifest["artifact_sha256"],
                       release_status="isolated_candidate_pending_integration_review" if all(confirmation.values()) else "held_failed_dev_confirmation")
        print("Dev confirmation:", report, flush=True)
    else:
        summary["release_status"] = "retain_current_model"
    if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != v for p, v in unchanged.items()):
        raise ValueError("Active artifact/config changed during fitting.")
    if provenance["source_sha256_lf"] != {p: sha256_file(ROOT / p) for p in sources}:
        raise ValueError("Experiment source changed during fitting.")
    _write(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/action_bilou_experiment.yaml")
    run(parser.parse_args(argv).config)


if __name__ == "__main__":
    main()
