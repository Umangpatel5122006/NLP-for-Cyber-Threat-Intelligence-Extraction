"""Compare fixed action CRFs with train-only grouped CV, then confirm on dev."""

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from cti.core.dataset import load_smah
from cti.eval.cv import get_report_grouped_splits
from cti.eval.metrics import evaluate_predictions
from cti.eval.report import generate_evaluation_summary
from cti.models.base import validate_training_data
from cti.models.crf import CRFTagger
from cti.models.ner_artifact import _environment, load_ner_tagger, save_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner


ROOT = Path(__file__).resolve().parents[2]
FEATURE_KEYS = {
    "window", "use_ortho", "use_affixes", "use_ioc_patterns",
    "use_gazetteer", "use_bigrams", "use_triggers",
}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or type(config.get("folds")) is not int or config["folds"] < 2:
        raise ValueError("Specify at least two report-grouped folds.")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer.")
    if type(config.get("minimum_cv_gain")) not in (int, float) or not 0 <= config["minimum_cv_gain"] <= 1:
        raise ValueError("minimum_cv_gain must lie between zero and one.")
    if type(config.get("minimum_fold_wins")) is not int or not 1 <= config["minimum_fold_wins"] <= config["folds"]:
        raise ValueError("minimum_fold_wins must lie within the fold count.")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Specify a fixed nonempty candidate list.")
    identifiers = set()
    allowed = set(inspect.signature(CRFTagger).parameters) - {"kwargs"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"id", "parameters"}:
            raise ValueError("Candidates must contain only id and classical CRF parameters.")
        identifier = candidate["id"]
        if not isinstance(identifier, str) or not identifier or not all(ch.isascii() and (ch.isalnum() or ch == "_") for ch in identifier):
            raise ValueError("Candidate IDs must contain ASCII letters, digits or underscores.")
        if identifier in identifiers:
            raise ValueError("Candidate IDs must be unique.")
        identifiers.add(identifier)
        params = candidate["parameters"]
        if not isinstance(params, dict) or set(params) - allowed:
            raise ValueError("Unsupported CRF parameters.")
        features = params.get("feature_config", {})
        if not isinstance(features, dict) or set(features) - FEATURE_KEYS:
            raise ValueError("Only fixed classical local-feature switches are supported.")
        for key, value in features.items():
            if key == "window":
                if type(value) is not int or not 0 <= value <= 5:
                    raise ValueError("Feature window must be an integer from zero to five.")
            elif type(value) is not bool:
                raise ValueError("Feature switches must be boolean.")
        CRFTagger(**params)
    if "baseline" not in identifiers:
        raise ValueError("Include a baseline candidate.")
    return config


def choose_candidate(reports, minimum_gain, minimum_fold_wins):
    baseline = next(report for report in reports if report["id"] == "baseline")
    baseline_f1 = baseline["pooled"]["exact_strict"]["f1"]
    eligible = []
    for report in reports:
        wins = sum(candidate["exact_strict"]["f1"] > reference["exact_strict"]["f1"]
                   for candidate, reference in zip(report["fold_metrics"], baseline["fold_metrics"]))
        gain = round(report["pooled"]["exact_strict"]["f1"] - baseline_f1, 4)
        if report["id"] != "baseline" and gain >= minimum_gain and wins >= minimum_fold_wins:
            eligible.append(report)
    return max(eligible, key=lambda report: report["pooled"]["exact_strict"]["f1"])["id"] if eligible else "baseline"


def _check_predictions(tokens, predictions):
    validate_training_data(tokens, predictions)
    if any(tag not in ("O", "B-ACTION", "I-ACTION") for sentence in predictions for tag in sentence):
        raise ValueError("Action predictions contain an out-of-layer tag.")


def _fit(candidate, data, indices):
    selected = [i for i in indices if data.relevant[i] and data.tokens[i]]
    if not selected:
        raise ValueError("A fold has no nonempty relevant training sentences.")
    model = CRFTagger(**candidate["parameters"])
    model.fit([data.tokens[i] for i in selected], [data.tags[i] for i in selected])
    return model


def run_experiment(config_path, raw_dir, splits_dir, processed_dir, model_dir, results_dir):
    config = read_config(config_path)
    model_dir, results_dir = Path(model_dir), Path(results_dir)
    if model_dir.resolve() == (ROOT / "models").resolve():
        raise ValueError("Use a separate experiment model directory.")
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ids = {split: set(json.loads((Path(splits_dir) / f"{split}_report_ids.json").read_text(encoding="utf-8")))
           for split in ("train", "dev", "test")}
    if ids["train"] & (ids["dev"] | ids["test"]) or ids["dev"] & ids["test"]:
        raise PermissionError("Official report splits must be disjoint.")
    # Dev rows are not selected until after CV selection is written.
    raw_train = load_smah(raw_dir=raw_dir, splits_dir=splits_dir, splits=["train"])["train"]
    train = load_aligned_ner("action", "train", raw_train, processed_dir)
    folds = get_report_grouped_splits(train.doc_ids, n_splits=config["folds"], seed=config["seed"])
    provenance = {
        "config_sha256_lf": sha256_file(config_path),
        "processed_train_sha256_lf": sha256_file(Path(processed_dir) / "bio_en_action_train.json"),
        "raw_dataset_sha256_lf": sha256_file(next(Path(raw_dir).rglob("SMAH-CTI.json"))),
        "source_sha256_lf": {name: sha256_file(ROOT / name) for name in (
            "src/cti/experiment_action.py", "src/cti/models/ner_data.py",
            "src/cti/eval/cv.py", "src/cti/eval/metrics.py",
        )},
        "environment": _environment(),
    }
    fingerprint = _digest(provenance)
    fold_reports = [{
        "train_report_ids": sorted({train.doc_ids[i] for i in fit}),
        "validation_report_ids": sorted({train.doc_ids[i] for i in validation}),
        "validation_sentences": len(validation),
    } for fit, validation in folds]
    reports = []
    for candidate in config["candidates"]:
        predictions = [None] * len(train.tokens)
        metrics = []
        for fold_index, (fit, validation) in enumerate(folds):
            cache = model_dir / f"{candidate['id']}_fold_{fold_index + 1}.json"
            signature = {"fingerprint": fingerprint, "candidate": candidate, "fold": fold_reports[fold_index]}
            cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
            tokens = [train.tokens[i] for i in validation]
            if cached.get("signature") == signature:
                decoded = cached["predictions"]
                if cached.get("predictions_sha256") != _digest(decoded):
                    raise ValueError(f"Cached predictions are corrupt: {cache}. Use a fresh experiment directory.")
                _check_predictions(tokens, decoded)
                print(f"{candidate['id']}: fold {fold_index + 1}/{len(folds)} resumed", flush=True)
            else:
                print(f"{candidate['id']}: fitting fold {fold_index + 1}/{len(folds)}", flush=True)
                model = _fit(candidate, train, fit)
                decoded = model.predict(tokens)
                _check_predictions(tokens, decoded)
                _write(cache, {"signature": signature, "predictions": decoded, "predictions_sha256": _digest(decoded)})
            for index, sequence in zip(validation, decoded):
                predictions[index] = sequence
            metrics.append(evaluate_predictions([train.tags[i] for i in validation], decoded))
            print(f"  all-sentence exact F1={metrics[-1]['exact_strict']['f1']:.4f}", flush=True)
            _write(results_dir / "progress.json", {"candidate": candidate["id"], "completed_fold": fold_index + 1})
        _check_predictions(train.tokens, predictions)
        reports.append({
            "id": candidate["id"], "parameters": candidate["parameters"],
            "fold_metrics": metrics, "pooled": evaluate_predictions(train.tags, predictions),
        })
        _write(results_dir / "cv.json", {"provenance": provenance, "folds": fold_reports, "candidates": reports})
    winner = choose_candidate(reports, config["minimum_cv_gain"], config["minimum_fold_wins"])
    selection = {
        "selected": winner, "criterion": "train-only pooled exact F1 with minimum gain and fold wins",
        "minimum_cv_gain": config["minimum_cv_gain"], "minimum_fold_wins": config["minimum_fold_wins"],
        "train_report_ids": sorted(set(train.doc_ids)), "dev_labels_used_for_selection": False,
        "test_split_used": False, "provenance": provenance,
    }
    _write(results_dir / "selection.json", selection)
    print(f"Selection frozen before dev evaluation: {winner}", flush=True)
    raw_dev = load_smah(raw_dir=raw_dir, splits_dir=splits_dir, splits=["dev"])["dev"]
    dev = load_aligned_ner("action", "dev", raw_dev, processed_dir)
    if set(train.uids) & set(dev.uids):
        raise ValueError("Train/dev sentence UIDs must be disjoint.")
    evaluations = {}
    for identifier in dict.fromkeys(["baseline", winner]):
        candidate = next(item for item in config["candidates"] if item["id"] == identifier)
        print(f"Fitting full-train model for dev confirmation: {identifier}", flush=True)
        model = _fit(candidate, train, range(len(train.tokens)))
        predictions = model.predict(dev.tokens)
        _check_predictions(dev.tokens, predictions)
        report = generate_evaluation_summary("crf_tagger", "action", dev.tags, predictions, dev.doc_ids, dev.sources, dev.relevant)
        artifact_path = model_dir / f"{identifier}_action.joblib"
        manifest = save_ner_tagger(model, artifact_path, "action", metadata={
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": selection, "training_policy": "relevant_only",
            "training_sentences": sum(bool(tokens) and relevant for tokens, relevant in zip(train.tokens, train.relevant)),
            "dev_report_ids": sorted(set(dev.doc_ids)), "test_split_used": False,
            "processed_dev_sha256_lf": sha256_file(Path(processed_dir) / "bio_en_action_dev.json"),
        })
        if load_ner_tagger(artifact_path, expected_layer="action").tagger.predict(dev.tokens) != predictions:
            raise RuntimeError("Saved experiment model changed dev predictions.")
        report.update(artifact=manifest, reload_predictions_identical=True)
        _write(results_dir / f"{identifier}_dev.json", report)
        _write(results_dir / f"{identifier}_dev.predictions.json", [
            {"uid": uid, "doc_id": doc_id, "tokens": tokens, "gold": gold, "predicted": predicted}
            for uid, doc_id, tokens, gold, predicted in zip(dev.uids, dev.doc_ids, dev.tokens, dev.tags, predictions)
        ])
        evaluations[identifier] = report
        print(f"{identifier}: all-dev exact F1={report['headline_all_sentences']['exact_strict']['f1']:.4f}; reload identical", flush=True)
    baseline_f1 = evaluations["baseline"]["headline_all_sentences"]["exact_strict"]["f1"]
    selected_f1 = evaluations[winner]["headline_all_sentences"]["exact_strict"]["f1"]
    result = {
        **selection, "baseline_dev_exact_f1": baseline_f1, "selected_dev_exact_f1": selected_f1,
        "dev_exact_gain": round(selected_f1 - baseline_f1, 4),
        "dev_improved": winner != "baseline" and selected_f1 > baseline_f1,
        "active_models_modified": False,
    }
    _write(results_dir / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/action_experiment.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/action_cv_v1")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/action_cv_v1")
    args = parser.parse_args(argv)
    run_experiment(args.config, args.raw_dir, args.splits_dir, args.processed_dir, args.model_dir, args.results_dir)


if __name__ == "__main__":
    main()
