"""Post-run BILOU cache/count audit; requires an actually completed experiment.

Preparation/encoding checks alone do not satisfy this audit. No CRF is refitted.
"""

import hashlib
import json
from pathlib import Path

import yaml

from audit_action_background import digest, recount, require
from cti.core.dataset import load_smah
from cti.core.sms_data import ROOT
from cti.eval.cv import get_report_grouped_splits
from cti.models.action_bilou_crf import load_candidate
from cti.models.ner_artifact import _environment, load_ner_tagger, sha256_file
from cti.models.ner_data import load_aligned_ner


def normalize_internal(tags):
    """Independent implementation of the frozen open-span prediction adapter."""
    output, previous = [], "O"
    for tag in tags:
        require(tag in {"O", "B-ACTION", "I-ACTION", "L-ACTION", "U-ACTION"}, "Unknown internal tag.")
        output.append("O" if tag == "O" else ("I-ACTION" if tag[0] in "IL" and previous[0] in "BI" else "B-ACTION"))
        previous = tag
    return output


def invalid_sequence(tags):
    state = "closed"
    for tag in tags:
        prefix = tag[0]
        if state == "closed" and prefix not in "OBU" or state == "open" and prefix not in "IL":
            return True
        state = "open" if prefix in "BI" else "closed"
    return state == "open"


def gates(base, candidate, policy):
    return {"f1_gain": candidate["f1"] >= base["f1"] + policy["minimum_f1_gain"],
            "exact_recall": candidate["recall"] >= base["recall"] - policy["maximum_exact_recall_loss"],
            "background_budget": candidate["background_predicted_spans"] <= base["background_predicted_spans"] * (1 + policy["maximum_background_increase"])}


def main():
    output, models = ROOT / "results/action_bilou_v1", ROOT / "models/action_bilou_v1"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    selection, summary, cv = read(output / "selection.json"), read(output / "summary.json"), read(output / "cv.json")
    config_path = ROOT / "configs/action_bilou_experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    p = selection["provenance"]
    require(p == summary["provenance"] == cv["provenance"] and p["environment"] == _environment(), "Runtime/provenance differs.")
    require(sha256_file(config_path) == p["config_sha256_lf"] and config["selection"] == selection["policy"], "Config changed.")
    for name, value in p["source_sha256_lf"].items():
        require(sha256_file(ROOT / name) == value, "Source changed: " + name)
    for name, value in p["unchanged_files_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value, "Active file changed: " + name)
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
    require(not ids["train"] & (ids["dev"] | ids["test"]) and not ids["dev"] & ids["test"], "Official split overlap.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    data = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    require(set(data.doc_ids) == ids["train"] == set(selection["train_report_ids"]), "Train reports differ.")
    baseline, candidate, internal = read(models / "baseline_oof.json"), read(models / "bilou_oof.json"), read(models / "internal_oof.json")
    require(len(baseline) == len(candidate) == len(internal) == len(data.uids), "OOF rows missing.")
    require(baseline == read(ROOT / "models/action_background_v1/relevant_only_oof.json"), "Reference predictions differ.")
    require([normalize_internal(tags) for tags in internal] == candidate, "Independent normalization differs.")
    reports = selection["candidates"]
    require([r["id"] for r in reports] == ["baseline", "bilou"] and reports == cv["candidates"], "Reports differ.")
    for guesses, report in zip((baseline, candidate), reports):
        require(recount(data, guesses, range(len(data.uids))) == report["pooled"], "Pooled strict counts differ.")
    folds = get_report_grouped_splits(data.doc_ids, n_splits=3, seed=42)
    require(len(folds) == len(cv["folds"]) == 3, "Fold count differs.")
    covered = []
    for n, ((fit, valid), record) in enumerate(zip(folds, cv["folds"]), 1):
        expected = {"train_report_ids": sorted({data.doc_ids[i] for i in fit}), "validation_report_ids": sorted({data.doc_ids[i] for i in valid}), "validation_sentences": len(valid)}
        require(record == expected and not set(record["train_report_ids"]) & set(record["validation_report_ids"]), "Fold leakage/mismatch.")
        fitted = [i for i in fit if data.tokens[i] and data.relevant[i]]
        signature = {"fingerprint": digest(p), "fold": record, "candidate": "bilou", "training": {"relevant_sentences": len(fitted), "uids_sha256": digest([data.uids[i] for i in fitted])}}
        cached = read(models / f"bilou_fold_{n}.json")
        require(cached["signature"] == signature and cached["internal_sha256"] == digest(cached["internal"])
                and cached["predictions_sha256"] == digest(cached["predictions"]), "Cache provenance/hash differs.")
        require(cached["internal"] == [internal[i] for i in valid] and cached["predictions"] == [candidate[i] for i in valid], "Cache/OOF differs.")
        for guesses, report in zip((baseline, candidate), reports):
            require(recount(data, guesses, valid) == report["folds"][n-1], "Fold counts differ.")
        covered.extend(valid)
    require(sorted(covered) == list(range(len(data.uids))), "Each train row must be held out once.")
    require(sum(invalid_sequence(tags) for tags in internal) == reports[1]["internal_invalid_sentences"], "Internal-invalid count differs.")
    checks = gates(reports[0]["pooled"], reports[1]["pooled"], config["selection"])
    wins = sum(a["f1"] > b["f1"] for a, b in zip(reports[1]["folds"], reports[0]["folds"]))
    passed = all(checks.values()) and wins >= config["selection"]["minimum_fold_wins"]
    require(checks == reports[1]["checks"] and wins == reports[1]["fold_wins"] and passed == reports[1]["eligible"], "Eligibility differs.")
    require(selection["selected"] == summary["selected"] == ("bilou" if passed else "baseline")
            and not selection["dev_used_for_selection"] and not selection["test_evaluated"] and not summary["test_evaluated"]
            and not summary["active_files_modified"], "Selection/scope differs.")
    if passed:
        require(summary["dev_evaluated"], "Dev confirmation missing.")
        raw_dev = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["dev"])["dev"]
        dev = load_aligned_ner("action", "dev", raw_dev, ROOT / "data/processed")
        rows = read(models / "dev_predictions.json")
        require(len(rows) == len(dev.uids) and set(dev.doc_ids) == ids["dev"] and not set(dev.uids) & set(data.uids), "Dev/train mismatch.")
        for i, row in enumerate(rows):
            require((row["uid"], row["report"], row["gold"], row["relevant"]) == (dev.uids[i], dev.doc_ids[i], dev.tags[i], dev.relevant[i]), "Dev alignment differs.")
            require(normalize_internal(row["internal"]) == row["candidate"], "Dev normalization differs.")
        before, after = [r["baseline"] for r in rows], [r["candidate"] for r in rows]
        a, b = recount(dev, before, range(len(rows))), recount(dev, after, range(len(rows)))
        report = read(output / "dev.json")
        require(report == summary["dev"] and report["baseline"] == a and report["candidate"] == b
                and report["checks"] == gates(a, b, config["confirmation"]), "Dev confirmation differs.")
        require(report["internal_invalid_sentences"] == sum(invalid_sequence(r["internal"]) for r in rows), "Dev internal count differs.")
        require(report["processed_dev_sha256_lf"] == sha256_file(ROOT / "data/processed/bio_en_action_dev.json"), "Dev BIO changed.")
        artifact, manifest = load_candidate(models / "candidate_action.joblib")
        require(artifact.predict(dev.tokens) == after and manifest["metadata"]["selection"] == selection
                and manifest["artifact_sha256"] == summary["candidate_artifact_sha256"], "Candidate reload/metadata differs.")
        require(load_ner_tagger(ROOT / "models/ner_action.joblib", expected_layer="action").tagger.predict(dev.tokens) == before, "Baseline dev differs.")
        require(all(report["checks"].values()) == summary["quantitative_checks_passed"], "Release decision differs.")
    else:
        require(not summary["dev_evaluated"] and not (output / "dev.json").exists() and not (models / "candidate_action.joblib").exists(), "Unexpected dev/candidate output.")
    result = {"passed": True, "scope": "Post-run independent cache/normalization/count audit; no independent CRF refits.",
              "selected": selection["selected"], "training_sentences": len(data.uids), "training_reports": len(set(data.doc_ids)),
              "fit_uid_hashes_verified": True, "active_files_unchanged": True, "dev_evaluated": summary["dev_evaluated"], "test_evaluated": False,
              "source_sha256_lf": {name: sha256_file(ROOT / name) for name in ("scripts/audit_action_bilou.py", "scripts/audit_action_background.py")}}
    (output / "integrity_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
