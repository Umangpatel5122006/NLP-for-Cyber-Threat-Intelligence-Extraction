"""Independently recount the rejected train-only action relevance experiment.

Uses saved local OOF predictions; never fits models or selects dev/test text.
Run from the project root after python -m cti.experiment_action_relevance.
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
from cti.models.ner_artifact import sha256_file
from cti.models.ner_data import load_aligned_ner


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def recount(rows, cutoff):
    gold = [row["gold"] for row in rows]
    predicted = [row["action"] if cutoff is None or row["score"] >= cutoff
                 else ["O"] * len(row["action"]) for row in rows]
    truth = Entities(gold, scheme=IOB2).entities
    guesses = Entities(predicted, scheme=IOB2).entities
    def spans(entities):
        return {(entity.tag, entity.start, entity.end) for entity in entities}
    tp = sum(len(spans(a) & spans(b)) for a, b in zip(truth, guesses))
    gold_count, predicted_count = sum(map(len, truth)), sum(map(len, guesses))
    relevant = sum(row["relevant"] for row in rows)
    skipped = sum(bool(cutoff is not None and row["score"] < cutoff and row["relevant"]) for row in rows)
    metrics = {"tp": tp, "gold_spans": gold_count, "predicted_spans": predicted_count,
               "precision": tp / predicted_count if predicted_count else 0.0,
               "recall": tp / gold_count if gold_count else 0.0,
               "f1": 2 * tp / (gold_count + predicted_count) if gold_count + predicted_count else 0.0,
               "background_predicted_spans": sum(len(spans(guess)) for row, guess in zip(rows, guesses) if not row["relevant"]),
               "relevant_sentences_skipped": skipped, "relevant_sentences": relevant,
               "relevant_sentence_skip_rate": skipped / relevant if relevant else 0.0}
    for key, function in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
        require(math.isclose(metrics[key], function(gold, predicted, mode="strict", scheme=IOB2, zero_division=0), abs_tol=1e-12),
                "Counts disagree with public seqeval metrics: " + key)
    return metrics


def main():
    output = ROOT / "results/action_relevance_v1"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    selection, summary = read(output / "selection.json"), read(output / "summary.json")
    config_path = ROOT / "configs/action_relevance_experiment.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    provenance = selection["provenance"]
    require(summary["provenance"] == provenance, "Selection/summary provenance differs.")
    require(sha256_file(config_path) == provenance["config_sha256_lf"], "Configuration changed.")
    for key in ("source_sha256_lf", "action_cache_sha256_lf"):
        for name, expected in provenance[key].items():
            require(sha256_file(ROOT / name) == expected, "Changed source/cache: " + name)
    for split, expected in provenance["split_sha256_lf"].items():
        require(sha256_file(ROOT / f"data/splits/{split}_report_ids.json") == expected, "Split IDs changed.")
    for name, key in (("data/raw/SMAH-CTI.json", "raw_sha256_lf"),
                      ("data/processed/bio_en_action_train.json", "processed_train_sha256_lf")):
        require(sha256_file(ROOT / name) == provenance[key], "Training corpus changed.")
    previous = read(ROOT / "results/indic_rules_v2/summary.json")["existing_model_artifact_sha256"]
    for name, expected in previous.items():
        path = ROOT / f"models/{name}.joblib"
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Existing model changed: " + name)
    require(previous["ner_action"] == provenance["active_action_sha256"], "Action artifact changed.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    data = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    rows_path = ROOT / "models/action_relevance_v1/oof.json"
    rows = read(rows_path)
    require(len(rows) == len(data.uids), "Incomplete OOF rows.")
    require(len(set(data.uids)) == len(data.uids), "Duplicate UIDs.")
    require(sorted(set(data.doc_ids)) == provenance["train_report_ids"], "Training report IDs differ.")
    for i, row in enumerate(rows):
        require((row["uid"], row["report"], row["gold"], row["relevant"]) ==
                (data.uids[i], data.doc_ids[i], data.tags[i], data.relevant[i]), "OOF alignment failed.")
        require(type(row["score"]) in (float, int) and math.isfinite(row["score"]) and 0 <= row["score"] <= 1,
                "Invalid relevance score.")
        require(len(row["action"]) == len(data.tokens[i]), "Token alignment failed.")
    folds = get_report_grouped_splits(data.doc_ids, n_splits=config["folds"], seed=config["seed"])
    protected = set(read(ROOT / "data/splits/dev_report_ids.json")) | set(read(ROOT / "data/splits/test_report_ids.json"))
    covered = []
    for number, (fit, valid) in enumerate(folds, 1):
        fit_ids, valid_ids = {data.doc_ids[i] for i in fit}, {data.doc_ids[i] for i in valid}
        require(not fit_ids & valid_ids and not (fit_ids | valid_ids) & protected, "Report leakage.")
        require(fit_ids | valid_ids == set(data.doc_ids), "Fold misses training reports.")
        cached = read(ROOT / f"models/action_features_v1/action_boundaries_fold_{number}.json")
        require(cached["predictions_sha256"] == digest(cached["predictions"]), "Cached prediction digest differs.")
        require(cached["signature"]["fold"] == {"train_report_ids": sorted(fit_ids), "validation_report_ids": sorted(valid_ids),
                                                   "validation_sentences": len(valid)}, "Cached fold differs.")
        require(cached["predictions"] == [rows[i]["action"] for i in valid], "OOF actions differ from held-out fold.")
        covered.extend(valid)
    require(sorted(covered) == list(range(len(rows))), "Each training row must be held out exactly once.")
    require([row["cutoff"] for row in selection["candidates"]] == [None, *config["cutoffs"]], "Cutoffs differ.")
    base = selection["candidates"][0]
    for candidate in selection["candidates"]:
        cutoff = candidate["cutoff"]
        require(recount(rows, cutoff) == candidate["pooled"], "Pooled counts differ.")
        require([recount([rows[i] for i in valid], cutoff) for _, valid in folds] == candidate["folds"], "Fold counts differ.")
        if cutoff is not None:
            policy = config["selection"]
            checks = {"f1_gain": candidate["pooled"]["f1"] >= base["pooled"]["f1"] + policy["minimum_f1_gain"],
                      "exact_recall": candidate["pooled"]["recall"] >= base["pooled"]["recall"] - policy["maximum_exact_recall_loss"],
                      "relevant_sentence_coverage": candidate["pooled"]["relevant_sentence_skip_rate"] <= policy["maximum_relevant_sentence_skip_rate"]}
            wins = sum(a["f1"] > b["f1"] for a, b in zip(candidate["folds"], base["folds"]))
            require(checks == candidate["checks"] and wins == candidate["fold_wins"], "Selection checks differ.")
            eligible = all(checks.values()) and wins >= policy["minimum_fold_wins"]
            require(eligible == candidate["eligible"] and not eligible, "Expected the frozen rejected experiment.")
    require(selection["selected"] == summary["selected"] == "no_gate" and selection["cutoff"] is summary["cutoff"] is None,
            "Rejected experiment must retain baseline.")
    require(not selection["dev_used_for_selection"] and not summary["dev_evaluated"] and
            not selection["test_evaluated"] and not summary["test_evaluated"] and not summary["active_files_modified"], "Scope flags differ.")
    require(not (output / "dev.json").exists() and not (ROOT / "models/action_relevance_v1/relevance.joblib").exists(),
            "Unexpected stale dev/model artifacts in the rejected experiment.")
    audit = {"passed": True, "training_sentences": len(rows), "training_reports": len(set(data.doc_ids)),
             "report_disjoint_folds": len(folds), "cutoffs_recounted": config["cutoffs"],
             "selected": "no_gate", "dev_evaluated": False, "test_evaluated": False,
             "existing_model_sha256_verified": previous, "oof_sha256_lf": sha256_file(rows_path),
             "independent_counts_match_public_seqeval": True,
             "scope": "Recounts cached train-only predictions; does not independently refit relevance models.",
             "audit_script_sha256_lf": sha256_file(Path(__file__))}
    (output / "integrity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
