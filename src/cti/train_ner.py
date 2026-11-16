"""Train, evaluate, save and reload classical NER on train/dev only."""

import argparse
import importlib
import inspect
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from cti.core.dataset import load_smah
from cti.core.model_policy import CLASSICAL_NER_MODULES, check_model_scope
from cti.core.registry import create
from cti.eval.report import export_report_csv, export_report_markdown, generate_evaluation_summary
from cti.models.base import validate_training_data
from cti.models.ner_artifact import load_ner_tagger, save_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner


ROOT = Path(__file__).resolve().parents[2]


def _release_prediction_handles(tagger):
    """Close temporary native CRFsuite readers before replacing fitted models.

    sklearn-crfsuite lazily caches this handle. Clearing it after close lets a
    subsequent prediction reopen the saved model; fitted parameters are intact.
    Other supported taggers have no such reader.
    """
    model = getattr(tagger, "model", None)
    handle = getattr(model, "_tagger", None)
    if handle is not None:
        handle.close()
        model._tagger = None


def train_from_config(config_path, raw_dir, splits_dir, processed_dir, model_dir,
                      results_dir, layers=("entities", "action"), run_name="ner_current"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name must be a filename prefix using letters, digits, underscores or hyphens.")
    if not layers or any(layer not in ("entities", "action") for layer in layers):
        raise ValueError("Select at least one supported NER layer.")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("layers"), dict):
        raise ValueError("NER training config must define layer mappings.")
    if type(config.get("train_relevant_only")) is not bool:
        raise ValueError("train_relevant_only must be explicitly true or false.")
    for layer in layers:
        settings = config["layers"].get(layer)
        if not isinstance(settings, dict) or not isinstance(settings.get("parameters", {}), dict):
            raise ValueError(f"Invalid training settings for layer {layer!r}.")
        name = settings.get("model")
        check_model_scope(name)
        if name not in CLASSICAL_NER_MODULES:
            raise ValueError(f"Unsupported classical NER model: {name!r}.")

    splits = load_smah(raw_dir=raw_dir, splits_dir=splits_dir, splits=["train", "dev"])
    train_rows, dev_rows = splits["train"], splits["dev"]
    train_ids, dev_ids = set(train_rows["doc_id"]), set(dev_rows["doc_id"])
    protected = set(json.loads((Path(splits_dir) / "test_report_ids.json").read_text(encoding="utf-8")))
    if (train_ids | dev_ids) & protected:
        raise PermissionError("Protected test reports cannot be used in NER development.")
    if train_ids & dev_ids or set(train_rows["uid"]) & set(dev_rows["uid"]):
        raise ValueError("Train/dev reports and sentence UIDs must not overlap.")
    datasets = {
        layer: (
            load_aligned_ner(layer, "train", train_rows, processed_dir),
            load_aligned_ner(layer, "dev", dev_rows, processed_dir),
        ) for layer in layers
    }
    for train, dev in datasets.values():
        if not train.tokens or not dev.tokens:
            raise ValueError("NER requires nonempty train and dev datasets.")
    raw_path = next(Path(raw_dir).rglob("SMAH-CTI.json"))
    git_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    reports = {}
    for layer, (train, dev) in datasets.items():
        settings = config["layers"][layer]
        name = settings["model"]
        importlib.import_module(CLASSICAL_NER_MODULES[name])
        tagger = create(name, **settings.get("parameters", {}))
        if name == "crf_tagger":
            from cti.models.action_crf import PROFILE_FLAGS, create_crf_tagger
            features = settings.get("parameters", {}).get("feature_config") or {}
            if any(features.get(flag, False) for flag in PROFILE_FLAGS):
                if layer != "action":
                    raise ValueError("Action feature profiles require the action layer.")
            tagger = create_crf_tagger(**settings.get("parameters", {}))
        unknown_parameters = set(settings.get("parameters", {})) - set(inspect.signature(type(tagger)).parameters)
        if unknown_parameters:
            raise ValueError(f"Unsupported {name} parameters: {sorted(unknown_parameters)}.")
        selected = [index for index, relevant in enumerate(train.relevant) if relevant or not config["train_relevant_only"]]
        train_tokens = [train.tokens[index] for index in selected]
        train_tags = [train.tags[index] for index in selected]
        if not train_tokens:
            raise ValueError("The selected NER training policy produced no training sentences.")
        print(f"Training {name}/{layer}: {len(selected)} train sentences; dev={len(dev.tokens)}", flush=True)
        tagger.fit(train_tokens, train_tags)
        predictions = tagger.predict(dev.tokens)
        validate_training_data(dev.tokens, predictions)
        summary = generate_evaluation_summary(
            model_name=name, layer=layer, gold=dev.tags, predictions=predictions,
            doc_ids=dev.doc_ids, sources=dev.sources, relevant_mask=dev.relevant,
        )
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_commit": git_result.stdout.strip() if git_result.returncode == 0 else None,
            "training_policy": "relevant_only" if config["train_relevant_only"] else "all_sentences",
            "training_sentences": len(selected), "dev_sentences": len(dev.tokens),
            "train_report_ids": sorted(train_ids), "dev_report_ids": sorted(dev_ids),
            "test_split_used": False,
            "raw_dataset_sha256_lf": sha256_file(raw_path),
            "processed_train_sha256_lf": sha256_file(Path(processed_dir) / f"bio_en_{layer}_train.json"),
            "processed_dev_sha256_lf": sha256_file(Path(processed_dir) / f"bio_en_{layer}_dev.json"),
            "training_config_sha256_lf": sha256_file(config_path),
            "source_sha256_lf": {path: sha256_file(ROOT / path) for path in (
                "src/cti/train_ner.py", "src/cti/models/ner_artifact.py",
                "src/cti/models/ner_data.py", "src/cti/eval/metrics.py",
            )},
            "metrics_version": summary["metrics_version"],
            "evaluation_scope": "original processed BIO; all-dev headline, relevant-dev diagnostic; no relevance gate",
        }
        model_path = Path(model_dir) / f"ner_{layer}.joblib"
        manifest = save_ner_tagger(tagger, model_path, layer, metadata=metadata)
        loaded = load_ner_tagger(model_path, expected_layer=layer)
        if loaded.tagger.predict(dev.tokens) != predictions:
            raise RuntimeError("Saved NER tagger changed dev predictions after reload.")
        summary["artifact"] = manifest
        summary["reload_predictions_identical"] = True
        prefix = Path(results_dir) / f"{run_name}_{layer}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        export_report_csv(summary, prefix.with_suffix(".csv"))
        export_report_markdown(summary, prefix.with_suffix(".md"))
        rows = [{"uid": uid, "doc_id": doc_id, "tokens": tokens, "gold": gold, "predicted": predicted}
                for uid, doc_id, tokens, gold, predicted in zip(dev.uids, dev.doc_ids, dev.tokens, dev.tags, predictions)]
        prefix.with_suffix(".predictions.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
        reports[layer] = summary
        exact = summary["headline_all_sentences"]["exact_default"]["f1"]
        print(f"{layer}: all-dev exact F1={exact:.4f}; reload identical; saved {model_path}", flush=True)
        # These are temporary fitted/reloaded objects, not serving models.
        # Release native readers deterministically between layers and at exit.
        _release_prediction_handles(loaded.tagger)
        _release_prediction_handles(tagger)
        del loaded, tagger
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ner_training.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--layer", choices=("entities", "action", "both"), default="both")
    parser.add_argument("--run-name", default="ner_current")
    args = parser.parse_args(argv)
    layers = ("entities", "action") if args.layer == "both" else (args.layer,)
    train_from_config(args.config, args.raw_dir, args.splits_dir, args.processed_dir,
                      args.model_dir, args.results_dir, layers, args.run_name)


if __name__ == "__main__":
    main()
