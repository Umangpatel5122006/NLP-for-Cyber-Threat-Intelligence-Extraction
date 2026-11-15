"""Independently recount the frozen action-ending experiment without refitting."""

import hashlib
import json
from pathlib import Path

import yaml

from audit_action_background import digest, recount, require
from cti.core.dataset import load_smah
from cti.core.sms_data import ROOT
from cti.eval.cv import get_report_grouped_splits
from cti.models.action_end_crf import load_candidate
from cti.models.ner_artifact import _environment, load_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner


def independent_checks(base, candidate, policy):
    return {"f1_gain": candidate["f1"] - base["f1"] >= policy["minimum_f1_gain"],
            "exact_recall": candidate["recall"] >= base["recall"] - policy["maximum_exact_recall_loss"],
            "background_budget": candidate["background_predicted_spans"] <= base["background_predicted_spans"] * (1 + policy["maximum_background_increase"])}


def main():
    output, models = ROOT / "results/action_ending_v1", ROOT / "models/action_ending_v1"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    selection, summary, cv = read(output / "selection.json"), read(output / "summary.json"), read(output / "cv.json")
    config_path = ROOT / "configs/action_ending_experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    provenance = selection["provenance"]
    require(provenance == summary["provenance"] == cv["provenance"] and provenance["environment"] == _environment(), "Provenance/runtime differs.")
    require(sha256_file(config_path) == provenance["config_sha256_lf"] and config["selection"] == selection["policy"], "Frozen config changed.")
    for name, expected in provenance["source_sha256_lf"].items():
        require(sha256_file(ROOT / name) == expected, "Experiment source changed: " + name)
    for name, expected in provenance["unchanged_files_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, "Active file changed: " + name)
    for name, key in (("results/action_background_v1/summary.json", "upstream_summary_sha256_lf"),
                      ("results/action_background_v1/integrity_audit.json", "upstream_audit_sha256_lf"),
                      ("models/action_background_v1/relevant_only_oof.json", "baseline_oof_sha256_lf"),
                      ("data/raw/SMAH-CTI.json", "raw_sha256_lf"), ("data/processed/bio_en_action_train.json", "processed_train_sha256_lf")):
        require(sha256_file(ROOT / name) == provenance[key], "Upstream reference/data changed: " + name)
    ids = {}
    for split, expected in provenance["split_sha256_lf"].items():
        path = ROOT / f"data/splits/{split}_report_ids.json"
        require(sha256_file(path) == expected, "Official split changed.")
        ids[split] = set(read(path))
    require(not ids["train"] & (ids["dev"] | ids["test"]) and not ids["dev"] & ids["test"], "Official split overlap.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    train = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    require(set(train.doc_ids) == ids["train"] == set(selection["train_report_ids"]), "Train reports differ.")
    folds = get_report_grouped_splits(train.doc_ids, n_splits=config["folds"], seed=config["seed"])
    require(len(folds) == len(cv["folds"]) == 3, "Fold count differs.")
    require([r["id"] for r in selection["candidates"]] == ["baseline", *[r["id"] for r in config["candidates"]]], "Candidate order differs.")
    reference, eligible, hashes, coverage = None, [], {}, []
    for report, old in zip(selection["candidates"], cv["candidates"]):
        require(all(report[k] == old[k] for k in old), "CV/selection differs.")
        name = report["id"]
        path = models / f"{name}_oof.json"
        guesses = read(path)
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        require(len(guesses) == len(train.uids), "Incomplete OOF predictions.")
        require(recount(train, guesses, range(len(train.uids))) == report["pooled"], "Pooled counts differ.")
        for n, ((fit_rows, valid), record) in enumerate(zip(folds, cv["folds"]), 1):
            expected = {"train_report_ids": sorted({train.doc_ids[i] for i in fit_rows}),
                        "validation_report_ids": sorted({train.doc_ids[i] for i in valid}), "validation_sentences": len(valid)}
            require(expected == record and not set(record["train_report_ids"]) & set(record["validation_report_ids"]), "Fold report overlap/mismatch.")
            require(recount(train, guesses, valid) == report["folds"][n-1], "Fold metrics differ.")
            if name == "baseline":
                coverage.extend(valid)
            else:
                candidate = next(row for row in config["candidates"] if row["id"] == name)
                fitted = [i for i in fit_rows if train.tokens[i] and train.relevant[i]]
                signature = {"fingerprint": digest(provenance), "candidate": candidate, "fold": record,
                             "training": {"relevant_sentences": len(fitted), "uids_sha256": digest([train.uids[i] for i in fitted])}}
                cached = read(models / f"{name}_fold_{n}.json")
                require(signature == cached["signature"] and cached["predictions_sha256"] == digest(cached["predictions"]), "Cache signature/hash differs.")
                require(cached["predictions"] == [guesses[i] for i in valid], "OOF differs from fold cache.")
        if reference is None:
            reference = report
            require(guesses == read(ROOT / "models/action_background_v1/relevant_only_oof.json"), "Baseline reference differs.")
        else:
            checks = independent_checks(reference["pooled"], report["pooled"], config["selection"])
            wins = sum(a["f1"] > b["f1"] for a, b in zip(report["folds"], reference["folds"]))
            passed = all(checks.values()) and wins >= config["selection"]["minimum_fold_wins"]
            require(checks == report["checks"] and wins == report["fold_wins"] and passed == report["eligible"], "Eligibility differs.")
            if passed:
                eligible.append(report)
    require(sorted(coverage) == list(range(len(train.uids))), "Each train row must be held out once.")
    winner = max(eligible, key=lambda r: r["pooled"]["f1"])["id"] if eligible else "baseline"
    require(winner == selection["selected"] == summary["selected"] and not selection["dev_used_for_selection"], "Selection differs.")
    require(not selection["test_evaluated"] and not summary["test_evaluated"] and not summary["active_files_modified"], "Scope differs.")
    if winner == "baseline":
        require(not summary["dev_evaluated"] and not (output / "dev.json").exists()
                and not (models / "candidate_action.joblib").exists(), "Unexpected dev/candidate artifacts.")
    else:
        require(summary["dev_evaluated"], "Missing dev confirmation.")
        raw_dev = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["dev"])["dev"]
        dev = load_aligned_ner("action", "dev", raw_dev, ROOT / "data/processed")
        require(set(dev.doc_ids) == ids["dev"] and not set(dev.uids) & set(train.uids), "Dev/train mismatch.")
        rows = read(models / "dev_predictions.json")
        require(len(rows) == len(dev.uids), "Missing dev predictions.")
        for i, row in enumerate(rows):
            require((row["uid"], row["report"], row["gold"], row["relevant"]) ==
                    (dev.uids[i], dev.doc_ids[i], dev.tags[i], dev.relevant[i]), "Dev UID/gold alignment differs.")
        a, b = [r["baseline"] for r in rows], [r["candidate"] for r in rows]
        before, after = recount(dev, a, range(len(rows))), recount(dev, b, range(len(rows)))
        checks = independent_checks(before, after, config["confirmation"])
        report = read(output / "dev.json")
        require(report == summary["dev"] and before == report["baseline"] and after == report["candidate"] and checks == report["checks"], "Dev counts differ.")
        require(report["processed_dev_sha256_lf"] == sha256_file(ROOT / "data/processed/bio_en_action_dev.json"), "Dev BIO changed.")
        artifact_path = models / "candidate_action.joblib"
        artifact = load_candidate(artifact_path)
        require(hashlib.sha256(artifact_path.read_bytes()).hexdigest() == summary["candidate_artifact_sha256"]
                and artifact.tagger.predict(dev.tokens) == b, "Candidate artifact/reloaded predictions differ.")
        require(load_ner_tagger(ROOT / "models/ner_action.joblib", expected_layer="action").tagger.predict(dev.tokens) == a, "Baseline dev predictions differ.")
        require(artifact.manifest["metadata"]["selection"] == selection and all(checks.values()) == summary["quantitative_checks_passed"], "Persisted selection/confirmation differs.")
    result = {"passed": True, "selected": winner, "training_sentences": len(train.uids), "training_reports": len(set(train.doc_ids)),
              "report_disjoint_folds": len(folds), "independent_strict_counts_match": True,
              "fit_uid_hashes_verified": True, "active_artifacts_and_configs_unchanged": True,
              "dev_evaluated": summary["dev_evaluated"], "test_evaluated": False, "oof_sha256_lf": hashes,
              "audit_script_sha256_lf": sha256_file(Path(__file__)),
              "shared_counter_sha256_lf": sha256_file(ROOT / "scripts/audit_action_background.py"),
              "scope": "Independent cache/count/selection audit; no independent CRF refit."}
    (output / "integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
