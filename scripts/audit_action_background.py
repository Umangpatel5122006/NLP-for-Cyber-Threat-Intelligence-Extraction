"""Recount the frozen action-background comparison without refitting models.

Run from the project root after python -m cti.experiment_action_background.
Dev is inspected only if the completed run already confirmed a train-CV winner.
"""

import hashlib
import json
import math
from pathlib import Path

import yaml
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import Entities, IOB2

from cti.core.dataset import load_smah
from cti.core.sms_data import ROOT
from cti.eval.cv import get_report_grouped_splits
from cti.models.ner_artifact import _environment, load_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def recount(data, predicted, rows):
    gold, guesses = [data.tags[i] for i in rows], [predicted[i] for i in rows]
    require(all(len(g) == len(p) for g, p in zip(gold, guesses)), "Token counts differ.")
    require(all(tag in ("O", "B-ACTION", "I-ACTION") for sentence in guesses for tag in sentence), "Invalid action tags.")
    truth_spans = [{(e.tag, e.start, e.end) for e in sentence} for sentence in Entities(gold, IOB2).entities]
    guess_spans = [{(e.tag, e.start, e.end) for e in sentence} for sentence in Entities(guesses, IOB2).entities]
    tp = sum(len(a & b) for a, b in zip(truth_spans, guess_spans))
    ng, np = sum(map(len, truth_spans)), sum(map(len, guess_spans))
    kwargs = dict(mode="strict", scheme=IOB2, zero_division=0)
    values = {"tp": tp, "gold_spans": ng, "predicted_spans": np,
              "precision": precision_score(gold, guesses, **kwargs), "recall": recall_score(gold, guesses, **kwargs),
              "f1": f1_score(gold, guesses, **kwargs),
              "background_predicted_spans": sum(len(spans) for i, spans in zip(rows, guess_spans) if not data.relevant[i]),
              "relevant_sentences_skipped": 0, "relevant_sentences": sum(data.relevant[i] for i in rows), "relevant_sentence_skip_rate": 0.0}
    # Public metrics and direct counts may differ by floating-point rounding.
    for key, expected in (("precision", tp / np if np else 0.0), ("recall", tp / ng if ng else 0.0),
                          ("f1", 2 * tp / (ng + np) if ng + np else 0.0)):
        require(math.isclose(values[key], expected, abs_tol=1e-12), "Public metric differs from counts: " + key)
    values["precision"] = tp / np if np else 0.0
    values["recall"] = tp / ng if ng else 0.0
    values["f1"] = 2 * tp / (ng + np) if ng + np else 0.0
    return values


def inventory(data, rows, policy, seed):
    positive = [i for i in rows if data.tokens[i] and data.relevant[i]]
    background = [i for i in rows if data.tokens[i] and not data.relevant[i]]
    require(all(all(t == "O" for t in data.tags[i]) for i in background), "Contradictory background annotations.")
    chosen = list(positive)
    for report in sorted({data.doc_ids[i] for i in background}):
        available = [i for i in background if data.doc_ids[i] == report]
        limit = sum(data.doc_ids[i] == report for i in positive) if policy == "capped_background" else len(available)
        available.sort(key=lambda i: (hashlib.sha256(f"{seed}:{data.uids[i]}".encode()).hexdigest(), data.uids[i]))
        chosen.extend(available[:limit])
    chosen.sort()
    return {"sentences": len(chosen), "relevant": len(positive), "background": len(chosen) - len(positive),
            "uids_sha256": digest([data.uids[i] for i in chosen])}


def policy_checks(base, candidate, policy):
    before, after = base["background_predicted_spans"], candidate["background_predicted_spans"]
    reduction = (before - after) / before if before else (0.0 if after == 0 else -1.0)
    return {"f1_gain": candidate["f1"] >= base["f1"] + policy["minimum_f1_gain"],
            "exact_recall": candidate["recall"] >= base["recall"] - policy["maximum_exact_recall_loss"],
            "background_reduction": reduction >= policy["minimum_background_reduction"]}


def main():
    output, models = ROOT / "results/action_background_v1", ROOT / "models/action_background_v1"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    selection, summary, cv = read(output / "selection.json"), read(output / "summary.json"), read(output / "cv.json")
    config_path = ROOT / "configs/action_background_experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    provenance = selection["provenance"]
    require(provenance == summary["provenance"] == cv["provenance"], "Provenance differs.")
    require(provenance["environment"] == _environment(), "Runtime/feature dependencies changed.")
    require(sha256_file(config_path) == provenance["config_sha256_lf"] and config["selection"] == selection["policy"], "Configuration changed.")
    for group in ("source_sha256_lf", "reference_cache_sha256_lf"):
        for name, expected in provenance[group].items():
            require(sha256_file(ROOT / name) == expected, "Source/cache changed: " + name)
    for name, expected in provenance["unchanged_files_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, "Active artifact/config changed: " + name)
    for name, key in (("data/raw/SMAH-CTI.json", "raw_sha256_lf"), ("data/processed/bio_en_action_train.json", "processed_train_sha256_lf")):
        require(sha256_file(ROOT / name) == provenance[key], "Training data changed.")
    ids = {}
    for split, expected in provenance["split_sha256_lf"].items():
        path = ROOT / f"data/splits/{split}_report_ids.json"
        require(sha256_file(path) == expected, "Official split changed.")
        ids[split] = set(read(path))
    require(not ids["train"] & (ids["dev"] | ids["test"]) and not ids["dev"] & ids["test"], "Official report overlap.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    data = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    require(set(data.doc_ids) == ids["train"] == set(selection["train_report_ids"]), "Training report mismatch.")
    folds = get_report_grouped_splits(data.doc_ids, n_splits=config["folds"], seed=config["seed"])
    require(len(folds) == len(cv["folds"]), "Fold count mismatch.")
    seen, baseline, eligible, oof_hashes = [], None, [], {}
    require([r["id"] for r in selection["candidates"]] == config["candidates"], "Candidate order changed.")
    for report, earlier in zip(selection["candidates"], cv["candidates"]):
        require(all(report[k] == earlier[k] for k in earlier), "CV/selection differs.")
        policy = report["id"]
        path = models / f"{policy}_oof.json"
        predictions = read(path)
        oof_hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        require(len(predictions) == len(data.uids), "Missing OOF sentences.")
        require(recount(data, predictions, range(len(data.tokens))) == report["pooled"], "Pooled count mismatch.")
        for n, ((fit, valid), record) in enumerate(zip(folds, cv["folds"]), 1):
            expected = {"train_report_ids": sorted({data.doc_ids[i] for i in fit}),
                        "validation_report_ids": sorted({data.doc_ids[i] for i in valid}), "validation_sentences": len(valid)}
            require(expected == record and not set(expected["train_report_ids"]) & set(expected["validation_report_ids"]), "Fold report mismatch/leakage.")
            require(recount(data, predictions, valid) == report["folds"][n-1], "Fold count mismatch.")
            if policy == "relevant_only":
                cache = read(ROOT / f"models/action_features_v1/action_boundaries_fold_{n}.json")
                seen.extend(valid)
            else:
                cache = read(models / f"{policy}_fold_{n}.json")
                counted = inventory(data, fit, policy, config["seed"])
                signature = {"fingerprint": digest(provenance), "policy": policy, "fold": record, "training": counted}
                require(counted == report["training"][n-1] and signature == cache["signature"], "Fit inventory/cache provenance mismatch.")
            require(cache["predictions_sha256"] == digest(cache["predictions"]), "Cache digest mismatch.")
            require(cache["predictions"] == [predictions[i] for i in valid], "OOF differs from held-out fold cache.")
        if baseline is None:
            baseline = report
        else:
            checks = policy_checks(baseline["pooled"], report["pooled"], config["selection"])
            wins = sum(a["f1"] > b["f1"] for a, b in zip(report["folds"], baseline["folds"]))
            passes = all(checks.values()) and wins >= config["selection"]["minimum_fold_wins"]
            require(checks == report["checks"] and wins == report["fold_wins"] and passes == report["eligible"], "Eligibility mismatch.")
            if passes:
                eligible.append(report)
    require(sorted(seen) == list(range(len(data.uids))), "OOF must cover each train sentence once.")
    selected = max(eligible, key=lambda r: r["pooled"]["f1"])["id"] if eligible else "relevant_only"
    require(selected == summary["selected"] == selection["selected"], "Winner mismatch.")
    require(not selection["dev_used_for_selection"] and not summary["test_evaluated"] and not selection["test_evaluated"]
            and not summary["active_files_modified"], "Scope mismatch.")
    if selected == "relevant_only":
        require(not summary["dev_evaluated"] and not (output / "dev.json").exists()
                and not (models / "candidate_action.joblib").exists(), "Unexpected dev/candidate artifacts.")
    else:
        require(summary["dev_evaluated"], "Missing dev confirmation.")
        raw_dev = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["dev"])["dev"]
        dev = load_aligned_ner("action", "dev", raw_dev, ROOT / "data/processed")
        require(set(dev.doc_ids) == ids["dev"] and not set(dev.uids) & set(data.uids), "Dev/train overlap.")
        rows = read(models / "dev_predictions.json")
        require(len(rows) == len(dev.uids), "Missing dev rows.")
        for i, row in enumerate(rows):
            require((row["uid"], row["report"], row["gold"], row["relevant"]) ==
                    (dev.uids[i], dev.doc_ids[i], dev.tags[i], dev.relevant[i]), "Dev alignment mismatch.")
        a, b = [r["baseline"] for r in rows], [r["candidate"] for r in rows]
        before, after = recount(dev, a, range(len(rows))), recount(dev, b, range(len(rows)))
        checks = policy_checks(before, after, config["confirmation"])
        saved = read(output / "dev.json")
        require(saved == summary["dev"] and saved["baseline"] == before and saved["candidate"] == after and saved["checks"] == checks, "Dev recount differs.")
        require(saved["processed_dev_sha256_lf"] == sha256_file(ROOT / "data/processed/bio_en_action_dev.json"), "Dev data changed.")
        artifact = load_ner_tagger(models / "candidate_action.joblib", expected_layer="action")
        require(hashlib.sha256((models / "candidate_action.joblib").read_bytes()).hexdigest() == summary["candidate_artifact_sha256"], "Candidate artifact changed.")
        require(artifact.tagger.predict(dev.tokens) == b, "Reload changed dev predictions.")
        require(inventory(data, range(len(data.uids)), selected, config["seed"]) == summary["training"] == artifact.manifest["metadata"]["training"], "Full training inventory mismatch.")
        require(all(checks.values()) == summary["quantitative_checks_passed"], "Dev decision mismatch.")
    result = {"passed": True, "selected": selected, "training_sentences": len(data.uids), "training_reports": len(set(data.doc_ids)),
              "report_disjoint_folds": len(folds), "candidate_policies_recounted": config["candidates"],
              "sampling_inventories_verified": True, "dev_evaluated": summary["dev_evaluated"], "test_evaluated": False,
              "active_artifacts_and_configs_unchanged": True, "oof_sha256_lf": oof_hashes,
              "audit_script_sha256_lf": sha256_file(Path(__file__)),
              "scope": "Independent cached-prediction recount and sampling audit; no independent CRF refits."}
    (output / "integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
