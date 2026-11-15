"""Audit frozen action OOF errors and source annotations on official train only.

Does not fit, score dev/test, change labels, or replace active artifacts.
Detailed source excerpts are written only to ignored local CSVs.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from cti.core.bio import load_label_map, to_bio
from cti.core.dataset import load_smah
from cti.core.sms_data import ROOT
from cti.eval.action_diagnostics import (STATUSES, action_spans, compare_gold, diagnose,
                                        duplicate_annotations, gold_statuses, overlap)
from cti.eval.cv import get_report_grouped_splits
from cti.experiment_action import _write
from cti.models.ner_artifact import _environment, sha256_file
from cti.models.ner_data import load_aligned_ner


def require(value, message):
    if not value:
        raise ValueError(message)


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run():
    output = ROOT / "results/action_error_audit_v1"
    output.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    previous = ROOT / "results/action_background_v1"
    frozen, audited = read(previous / "summary.json"), read(previous / "integrity_audit.json")
    provenance = frozen["provenance"]
    require(audited["passed"] and audited["audit_script_sha256_lf"] == sha256_file(ROOT / "scripts/audit_action_background.py"),
            "Run the background experiment's integrity audit first.")
    require(provenance["environment"] == _environment(), "Upstream runtime/feature resources changed.")
    for key in ("source_sha256_lf", "reference_cache_sha256_lf"):
        for name, expected in provenance[key].items():
            require(sha256_file(ROOT / name) == expected, "Upstream source/cache changed: " + name)
    for name, expected in provenance["unchanged_files_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, "Active file changed: " + name)
    for name, expected in audited["oof_sha256_lf"].items():
        require(sha256_file(ROOT / name) == expected, "Audited OOF changed: " + name)
    for split, expected in provenance["split_sha256_lf"].items():
        require(sha256_file(ROOT / f"data/splits/{split}_report_ids.json") == expected, "Official split changed.")
    require(sha256_file(ROOT / "data/raw/SMAH-CTI.json") == provenance["raw_sha256_lf"]
            and sha256_file(ROOT / "data/processed/bio_en_action_train.json") == provenance["processed_train_sha256_lf"], "Training corpus changed.")
    selection = read(previous / "selection.json")
    require(selection["provenance"] == provenance and not selection["dev_used_for_selection"], "Upstream selection differs.")
    require(sha256_file(ROOT / "configs/action_background_experiment.yaml") == provenance["config_sha256_lf"], "Upstream config changed.")
    dependencies = ("scripts/build_action_error_audit.py", "src/cti/eval/action_diagnostics.py", "src/cti/eval/metrics.py",
                    "src/cti/core/bio.py", "src/cti/preprocessing/tokenizer.py", "configs/labels.yaml")
    source_hashes = {name: sha256_file(ROOT / name) for name in dependencies}
    print("Loading official train only; validating frozen OOF inputs", flush=True)
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    data = load_aligned_ner("action", "train", raw, ROOT / "data/processed")
    raw_by_uid = {row["uid"]: row for row in raw.to_dict("records")}
    token_rows = {row["uid"]: row["tokens"] for row in read(ROOT / "data/processed/bio_en_action_train.json")}
    require(set(data.doc_ids) == set(selection["train_report_ids"]), "Training reports differ.")
    folds = get_report_grouped_splits(data.doc_ids, n_splits=3, seed=42)
    cv = read(previous / "cv.json")
    for (fit, valid), record in zip(folds, cv["folds"]):
        require(set(record["train_report_ids"]) == {data.doc_ids[i] for i in fit}
                and set(record["validation_report_ids"]) == {data.doc_ids[i] for i in valid}, "Fold alignment differs.")
    predictions, reports, per_report = {}, {}, {}
    for row in selection["candidates"]:
        policy = row["id"]
        guessed = read(ROOT / f"models/action_background_v1/{policy}_oof.json")
        report = diagnose(data.tags, guessed, data.relevant)
        for name, value in report["exact_strict"].items():
            require(value == row["pooled"][name], "Strict OOF counts differ: " + policy)
        require(report["taxonomy_exact_first"]["background_predicted_spans"] == row["pooled"]["background_predicted_spans"], "Background count differs.")
        predictions[policy], reports[policy] = guessed, report
        per_report[policy] = {}
        for doc_id in sorted(set(data.doc_ids)):
            indices = [i for i, doc in enumerate(data.doc_ids) if doc == doc_id]
            details = diagnose([data.tags[i] for i in indices], [guessed[i] for i in indices], [data.relevant[i] for i in indices])
            per_report[policy][doc_id] = {**details["exact_strict"],
                "boundary_pairs": details["taxonomy_exact_first"]["boundary_pairs"], "no_overlap_gold": details["gold_any_overlap_status"]["no_overlap"],
                "background_predicted_spans": details["taxonomy_exact_first"]["background_predicted_spans"]}
    transitions = {policy: compare_gold(data.tags, predictions["relevant_only"], guessed)
                   for policy, guessed in predictions.items() if policy != "relevant_only"}
    # Lexical exposure is fitted/report-specific, never counted against all train.
    lexical = {policy: {key: Counter({status: 0 for status in STATUSES}) for key in ("seen_phrase", "unseen_phrase")} for policy in predictions}
    fold_sentence_overlap = []
    for fit, valid in folds:
        fit_tags = defaultdict(set)
        for i in fit:
            fit_tags[tuple(data.tokens[i])].add(tuple(data.tags[i]))
        exposure = Counter({"validation_sentences": len(valid), "identical_tokens_in_fit": 0,
                            "same_tags_in_fit": 0, "conflicting_tags_in_fit": 0})
        for i in valid:
            seen_tags = fit_tags.get(tuple(data.tokens[i]), set())
            exposure["identical_tokens_in_fit"] += bool(seen_tags)
            exposure["same_tags_in_fit"] += tuple(data.tags[i]) in seen_tags
            exposure["conflicting_tags_in_fit"] += any(tags != tuple(data.tags[i]) for tags in seen_tags)
        fold_sentence_overlap.append(dict(exposure))
        phrases = {tuple(word.casefold() for word in data.tokens[i][g[1]:g[2]])
                   for i in fit if data.relevant[i] for g in action_spans(data.tags[i])}
        for i in valid:
            gs = action_spans(data.tags[i])
            for policy, guessed in predictions.items():
                for g, status in gold_statuses(gs, action_spans(guessed[i])):
                    phrase = tuple(word.casefold() for word in data.tokens[i][g[1]:g[2]])
                    lexical[policy]["seen_phrase" if phrase in phrases else "unseen_phrase"][status] += 1
    print("Checking source annotations and regenerating train BIO in memory", flush=True)
    annotation_counts, annotation_rows = Counter(), []
    label_map = load_label_map(ROOT / "configs/labels.yaml")
    def issue(kind, uid, annotation="", source="", group=""):
        if kind == "duplicate_action_tag_conflict" or sum(row["issue"] == kind for row in annotation_rows) < 12:
            annotation_rows.append({"issue": kind, "uid": uid, "report": raw_by_uid[uid]["doc_id"],
                                    "conflict_group_sha256": group,
                                    "context_before": raw_by_uid[uid].get("context_before", ""),
                                    "annotation": annotation, "source": source, "review_notes": ""})
    for uid in sorted(raw_by_uid):
        row = raw_by_uid[uid]
        text, entities = row["sentence_text"], row.get("entities") or []
        for entity in entities:
            if entity["label"] != "Action":
                continue
            annotation_counts["raw_action_annotations"] += 1
            start, end = entity.get("start"), entity.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                annotation_counts["invalid_action_offsets"] += 1
                issue("invalid_action_offsets", uid, str(entity), text)
            elif entity.get("text") != text[start:end]:
                annotation_counts["action_text_slice_mismatches"] += 1
                issue("action_text_slice_mismatch", uid, str(entity.get("text")), text[start:end])
        try:
            regenerated = to_bio(text, entities, label_map, keep={"ACTION"})
        except (ValueError, TypeError) as exc:
            annotation_counts["bio_regeneration_errors"] += 1
            issue("bio_regeneration_error", uid, str(exc), text)
            continue
        original = [tuple(token) for token in token_rows[uid]]
        projected = []
        for entity in entities:
            if entity["label"] == "Action" and type(entity.get("start")) is int and type(entity.get("end")) is int and 0 <= entity["start"] < entity["end"] <= len(text):
                single = to_bio(text, [entity], label_map, keep={"ACTION"})
                spans = action_spans([t[3] for t in single])
                annotation_counts["raw_annotations_without_projected_tokens"] += int(not spans)
                projected.extend(spans)
        retained = set(action_spans([t[3] for t in regenerated]))
        annotation_counts["duplicate_projected_action_spans"] += len(projected) - len(set(projected))
        annotation_counts["unique_projected_spans_not_retained"] += len(set(projected) - retained)
        annotation_counts["removed_projected_spans_overlapping_retained"] += sum(any(overlap(g, p) for p in retained) for g in set(projected) - retained)
        annotation_counts["retained_spans_not_from_single_annotations"] += len(retained - set(projected))
        if regenerated != original:
            annotation_counts["bio_regeneration_mismatched_sentences"] += 1
            if [t[:3] for t in regenerated] != [t[:3] for t in original]:
                annotation_counts["tokenization_mismatched_sentences"] += 1
            issue("bio_regeneration_mismatch", uid, json.dumps([t[3] for t in regenerated]), text)
    for key in ("raw_action_annotations", "invalid_action_offsets", "action_text_slice_mismatches", "bio_regeneration_errors",
                "bio_regeneration_mismatched_sentences", "tokenization_mismatched_sentences", "raw_annotations_without_projected_tokens",
                "duplicate_projected_action_spans", "unique_projected_spans_not_retained", "retained_spans_not_from_single_annotations"):
        annotation_counts[key] += 0
    if not annotation_counts["bio_regeneration_errors"]:
        require(annotation_counts["raw_action_annotations"] - annotation_counts["raw_annotations_without_projected_tokens"]
                - annotation_counts["duplicate_projected_action_spans"] - annotation_counts["unique_projected_spans_not_retained"]
                + annotation_counts["retained_spans_not_from_single_annotations"] == reports["relevant_only"]["exact_strict"]["gold_spans"],
                "Raw annotation projection counts do not reconcile with frozen gold.")
    duplicates = duplicate_annotations(data.tokens, data.tags, data.relevant, data.doc_ids)
    grouped = defaultdict(list)
    for i, words in enumerate(data.tokens):
        if words:
            grouped[tuple(words)].append(i)
    for indices in grouped.values():
        if len({tuple(data.tags[i]) for i in indices}) > 1:
            group = hashlib.sha256(json.dumps(data.tokens[indices[0]]).encode()).hexdigest()
            for i in sorted(indices, key=lambda n: data.uids[n]):
                issue("duplicate_action_tag_conflict", data.uids[i], json.dumps(data.tags[i]), raw_by_uid[data.uids[i]]["sentence_text"], group)
    # Stable, bounded source examples for human review; never a new evaluation set.
    examples, example_counts = [], Counter()
    ordered = sorted(range(len(data.uids)), key=lambda i: hashlib.sha256(data.uids[i].encode()).hexdigest())
    for i in ordered:
        uid, gs = data.uids[i], action_spans(data.tags[i])
        statuses = {policy: dict(gold_statuses(gs, action_spans(guessed[i]))) for policy, guessed in predictions.items()}
        for g in gs:
            categories = []
            if statuses["relevant_only"][g] != "exact":
                categories.append("baseline_" + statuses["relevant_only"][g])
            for policy in ("capped_background", "all_background"):
                if statuses["relevant_only"][g] == "exact" and statuses[policy][g] != "exact":
                    categories.append(policy + "_lost_exact")
            start, end = token_rows[uid][g[1]][1], token_rows[uid][g[2]-1][2]
            for category in categories:
                if example_counts[category] >= 12:
                    continue
                example_counts[category] += 1
                text = raw_by_uid[uid]["sentence_text"]
                example = {"category": category, "uid": uid, "report": data.doc_ids[i], "source_text": text,
                           "gold_start": start, "gold_end": end, "gold_text": text[start:end]}
                for policy, guessed in predictions.items():
                    example[policy + "_spans"] = json.dumps([text[token_rows[uid][p[1]][1]:token_rows[uid][p[2]-1][2]]
                                                             for p in action_spans(guessed[i])], ensure_ascii=False)
                example["review_notes"] = ""
                examples.append(example)
    write_csv(output / "review_examples.csv", examples, ["category", "uid", "report", "source_text", "gold_start", "gold_end", "gold_text",
              "relevant_only_spans", "capped_background_spans", "all_background_spans", "review_notes"])
    write_csv(output / "annotation_review.csv", annotation_rows, ["issue", "conflict_group_sha256", "uid", "report", "context_before", "annotation", "source", "review_notes"])
    # Integrity guards cover both original experiment inputs and audit code.
    for name, expected in provenance["unchanged_files_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, "Active file changed during audit.")
    require(source_hashes == {name: sha256_file(ROOT / name) for name in dependencies}, "Audit source changed during run.")
    summary = {"scope": "Descriptive train-only audit of previously selected report-grouped OOF predictions; not independent evaluation.",
               "training_reports": len(set(data.doc_ids)), "training_sentences": len(data.uids),
               "policies": reports, "per_report": per_report, "gold_status_transitions_from_baseline": transitions,
               "fold_fit_phrase_exposure": lexical, "source_annotation_checks": dict(annotation_counts),
               "fold_sentence_overlap": fold_sentence_overlap,
               "duplicate_annotations": duplicates, "local_review_example_counts": dict(example_counts),
               "dev_evaluated": False, "test_evaluated": False, "model_weights_changed": False, "labels_modified": False,
               "source_sha256_lf": source_hashes, "oof_sha256_lf": audited["oof_sha256_lf"],
               "raw_sha256_lf": provenance["raw_sha256_lf"], "processed_train_sha256_lf": provenance["processed_train_sha256_lf"],
               "upstream_summary_sha256_lf": sha256_file(previous / "summary.json"),
               "upstream_audit_sha256_lf": sha256_file(previous / "integrity_audit.json")}
    _write(output / "summary.json", summary)
    _write(output / "integrity_audit.json", {"passed": True, "strict_counts_reproduce_frozen_cv": True,
           "all_policy_gold_counts_identical": len({r["exact_strict"]["gold_spans"] for r in reports.values()}) == 1,
           "upstream_inputs_match_audited_hashes": True, "active_files_unchanged": True,
           "source_annotation_checks": dict(annotation_counts), "dev_evaluated": False, "test_evaluated": False,
           "summary_sha256_lf": sha256_file(output / "summary.json"),
           "note": "Integrity pass is not an annotation-quality approval; review reported source/duplicate discrepancies separately."})
    print(json.dumps({"annotation_checks": dict(annotation_counts), "duplicates": duplicates,
                      "taxonomy": {p: r["taxonomy_exact_first"] for p, r in reports.items()}, "transitions": transitions}, indent=2))
    return summary


if __name__ == "__main__":
    run()
