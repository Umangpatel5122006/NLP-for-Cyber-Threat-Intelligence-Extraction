"""Independent fit-inventory/count audit of the completed consistency experiment."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

from audit_action_background import digest, recount, require
from cti.core.dataset import load_smah
from cti.core.sms_data import ROOT
from cti.eval.cv import get_report_grouped_splits
from cti.models.ner_artifact import _environment, load_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner


def inventory(data, indices, policy):
    fit = [i for i in indices if data.relevant[i] and data.tokens[i]]
    labels, members = defaultdict(set), defaultdict(list)
    for i in fit:
        key = tuple(data.tokens[i])
        labels[key].add(tuple(data.tags[i])); members[key].append(i)
    ambiguous = sorted(i for i in fit if len(labels[tuple(data.tokens[i])]) > 1)
    consistent = [i for i in fit if i not in set(ambiguous)]
    kept = sorted(consistent if policy == "consistent_inputs" else [min(rows, key=lambda i: data.uids[i])
                  for key, rows in members.items() if len(labels[key]) == 1])
    dropped_duplicates = sorted(set(consistent) - set(kept))
    require(set(kept) <= set(fit) and not set(kept) & set(ambiguous), "Invalid fit inventory.")
    return {"input_relevant_sentences": len(fit), "selected_sentences": len(kept),
            "ambiguous_groups": sum(len(tags) > 1 for tags in labels.values()), "ambiguous_rows_removed": len(ambiguous),
            "duplicate_rows_removed": len(dropped_duplicates), "selected_uids_sha256": digest([data.uids[i] for i in kept]),
            "ambiguous_uids_sha256": digest([data.uids[i] for i in ambiguous]),
            "duplicate_uids_sha256": digest([data.uids[i] for i in dropped_duplicates])}


def checks(base, candidate, policy):
    return {"f1_gain": candidate["f1"] >= base["f1"] + policy["minimum_f1_gain"],
            "exact_recall": candidate["recall"] >= base["recall"] - policy["maximum_exact_recall_loss"],
            "background_budget": candidate["background_predicted_spans"] <= base["background_predicted_spans"] * (1 + policy["maximum_background_increase"])}


def main():
    output, models = ROOT / "results/action_consistency_v1", ROOT / "models/action_consistency_v1"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    selection, summary, cv = read(output / "selection.json"), read(output / "summary.json"), read(output / "cv.json")
    config_path = ROOT / "configs/action_consistency_experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    p = selection["provenance"]
    require(p == summary["provenance"] == cv["provenance"] and p["environment"] == _environment(), "Provenance/runtime differs.")
    require(sha256_file(config_path) == p["config_sha256_lf"] and config["selection"] == selection["policy"], "Config changed.")
    for name, expected in p["source_sha256_lf"].items():
        require(sha256_file(ROOT / name) == expected, "Source changed: " + name)
    for name, expected in p["unchanged_files_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, "Active file changed: " + name)
    for name, key in (("data/raw/SMAH-CTI.json", "raw_sha256_lf"), ("data/processed/bio_en_action_train.json", "processed_train_sha256_lf"),
                      ("models/action_background_v1/relevant_only_oof.json", "baseline_oof_sha256_lf"),
                      ("results/action_background_v1/summary.json", "upstream_summary_sha256_lf"),
                      ("results/action_background_v1/integrity_audit.json", "upstream_audit_sha256_lf")):
        require(sha256_file(ROOT / name) == p[key], "Reference/data changed: " + name)
    ids = {}
    for split, expected in p["split_sha256_lf"].items():
        path = ROOT / f"data/splits/{split}_report_ids.json"
        require(sha256_file(path) == expected, "Official split changed.")
        ids[split] = set(read(path))
    require(not ids["train"] & (ids["dev"] | ids["test"]) and not ids["dev"] & ids["test"], "Split overlap.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    data = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    require(set(data.doc_ids) == ids["train"] == set(selection["train_report_ids"]), "Train reports differ.")
    folds = get_report_grouped_splits(data.doc_ids, n_splits=3, seed=42)
    require(len(folds) == len(cv["folds"]) == 3, "Fold count differs.")
    require([row["id"] for row in selection["candidates"]] == ["baseline", *config["candidates"]], "Candidate order differs.")
    base, eligible, hashes, coverage = None, [], {}, []
    for report, old in zip(selection["candidates"], cv["candidates"]):
        require(all(report[k] == old[k] for k in old), "CV/selection differs.")
        name = report["id"]
        path = models / f"{name}_oof.json"
        predictions = read(path)
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        require(len(predictions) == len(data.uids) and recount(data, predictions, range(len(data.uids))) == report["pooled"], "Pooled original-gold counts differ.")
        for n, ((fit, valid), record) in enumerate(zip(folds, cv["folds"]), 1):
            expected = {"train_report_ids": sorted({data.doc_ids[i] for i in fit}), "validation_report_ids": sorted({data.doc_ids[i] for i in valid}), "validation_sentences": len(valid)}
            require(record == expected and not set(record["train_report_ids"]) & set(record["validation_report_ids"]), "Report overlap/fold mismatch.")
            require(recount(data, predictions, valid) == report["folds"][n-1], "Fold original-gold counts differ.")
            if name == "baseline":
                coverage.extend(valid)
            else:
                selected = inventory(data, fit, name)
                cached = read(models / f"{name}_fold_{n}.json")
                signature = {"fingerprint": digest(p), "policy": name, "fold": record, "training": selected}
                require(selected == report["training"][n-1] and cached["signature"] == signature, "Fit-only selection inventory differs.")
                require(cached["predictions_sha256"] == digest(cached["predictions"])
                        and cached["predictions"] == [predictions[i] for i in valid], "Cache/predictions differ.")
        if base is None:
            base = report
            require(predictions == read(ROOT / "models/action_background_v1/relevant_only_oof.json"), "Baseline differs.")
        else:
            gates = checks(base["pooled"], report["pooled"], config["selection"])
            wins = sum(a["f1"] > b["f1"] for a, b in zip(report["folds"], base["folds"]))
            allowed = all(gates.values()) and wins >= config["selection"]["minimum_fold_wins"]
            require(gates == report["checks"] and wins == report["fold_wins"] and allowed == report["eligible"], "Eligibility differs.")
            if allowed:
                eligible.append(report)
    require(sorted(coverage) == list(range(len(data.uids))), "Validation must retain every train row exactly once.")
    winner = max(eligible, key=lambda r: r["pooled"]["f1"])["id"] if eligible else "baseline"
    require(winner == selection["selected"] == summary["selected"] and not selection["dev_used_for_selection"]
            and not selection["test_evaluated"] and not summary["test_evaluated"] and not summary["active_files_modified"]
            and not selection["gold_labels_modified"] and not summary["gold_labels_modified"]
            and not selection["validation_filtered"] and not summary["validation_filtered"], "Winner or scope differs.")
    if winner == "baseline":
        require(not summary["dev_evaluated"] and not (output / "dev.json").exists() and not (models / "candidate_action.joblib").exists(), "Unexpected dev/model output.")
    else:
        require(summary["dev_evaluated"], "Missing dev confirmation.")
        raw_dev = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["dev"])["dev"]
        dev = load_aligned_ner("action", "dev", raw_dev, ROOT / "data/processed")
        rows = read(models / "dev_predictions.json")
        require(len(rows) == len(dev.uids) and set(dev.doc_ids) == ids["dev"] and not set(dev.uids) & set(data.uids), "Dev/train mismatch.")
        for i, row in enumerate(rows):
            require((row["uid"], row["report"], row["gold"], row["relevant"]) == (dev.uids[i], dev.doc_ids[i], dev.tags[i], dev.relevant[i]), "Dev alignment differs.")
        before, after = [row["baseline"] for row in rows], [row["candidate"] for row in rows]
        report = read(output / "dev.json")
        a, b = recount(dev, before, range(len(rows))), recount(dev, after, range(len(rows)))
        require(report == summary["dev"] and report["baseline"] == a and report["candidate"] == b
                and report["checks"] == checks(a, b, config["confirmation"]), "Dev counts/decision differ.")
        require(sha256_file(ROOT / "data/processed/bio_en_action_dev.json") == report["processed_dev_sha256_lf"], "Dev data changed.")
        artifact_path = models / "candidate_action.joblib"
        artifact = load_ner_tagger(artifact_path, expected_layer="action")
        require(hashlib.sha256(artifact_path.read_bytes()).hexdigest() == summary["candidate_artifact_sha256"]
                and artifact.tagger.predict(dev.tokens) == after, "Candidate artifact/reload differs.")
        require(artifact.manifest["metadata"]["selection"] == selection and artifact.manifest["metadata"]["training"] == summary["training"]
                == inventory(data, range(len(data.uids)), winner), "Full fit inventory differs.")
        require(load_ner_tagger(ROOT / "models/ner_action.joblib", expected_layer="action").tagger.predict(dev.tokens) == before
                and all(report["checks"].values()) == summary["quantitative_checks_passed"], "Baseline/release decision differs.")
    result = {"passed": True, "scope": "Independent fit-inventory and cached-count audit; no independent CRF refits.",
              "selected": winner, "training_sentences": len(data.uids), "training_reports": len(set(data.doc_ids)),
              "fit_only_selection_verified": True, "original_validation_retained": True, "gold_labels_modified": False,
              "dev_evaluated": summary["dev_evaluated"], "test_evaluated": False, "active_files_unchanged": True,
              "full_train_policy_inventories": {policy: inventory(data, range(len(data.uids)), policy) for policy in config["candidates"]},
              "oof_sha256_lf": hashes, "source_sha256_lf": {name: sha256_file(ROOT / name) for name in ("scripts/audit_action_consistency.py", "scripts/audit_action_background.py")}}
    (output / "integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
