"""Train-only annotation review pack; never fits models or rewrites labels."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

from cti.core.dataset import load_smah
from cti.core.sms_data import ROOT
from cti.eval.action_diagnostics import action_spans, overlap
from cti.models.ner_artifact import sha256_file
from cti.models.ner_data import load_aligned_ner


DECISIONS = frozenset(("pending", "context_explains", "inconsistent_labels", "source_extraction_issue", "uncertain"))
FIELDS = ("group_id", "evidence_sha256", "difference", "identical_context_conflict", "decision", "reviewer", "reviewed_on", "notes")
QUESTIONS = {
    "presence_absence": "Should this occurrence contain an action? Compare relevance, body text, captions and navigation before deciding.",
    "different_actions": "Which behavior is the target? The variants select different, non-overlapping actions.",
    "span_count_or_segmentation": "Should coordinated behaviors be separate spans? Compare command arguments, purpose clauses and omitted actions.",
    "boundary_extent": "Which words belong to the same action? Compare auxiliaries, modifiers, objects, paths and prepositional arguments.",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def difference(variants):
    if any(not spans for spans in variants):
        return "presence_absence"
    if any(not any(overlap(g, p) for g in a for p in b) for a, b in combinations(variants, 2)):
        return "different_actions"
    if len({len(spans) for spans in variants}) > 1:
        return "span_count_or_segmentation"
    return "boundary_extent"


def minimum_disagreements(rows, with_context=False):
    """Sentence-exact lower bound for one fixed deterministic token-only tagger.

    This is not a span-F1 ceiling or an estimate of model-training damage.
    """
    inputs = defaultdict(Counter)
    for row in rows:
        key = (tuple(row["words"]), row["context_before"] if with_context else "")
        inputs[key][tuple(row["tags"])] += 1
    return sum(sum(counts.values()) - max(counts.values()) for counts in inputs.values())


def build_groups(data, raw_rows):
    raw = {row["uid"]: row for row in raw_rows}
    if len(raw) != len(raw_rows) or set(raw) != set(data.uids):
        raise ValueError("Review data must have complete, unique raw/BIO UID alignment.")
    by_tokens = defaultdict(list)
    for i, uid in enumerate(data.uids):
        if data.tokens[i]:
            spans = action_spans(data.tags[i])
            source = raw[uid]
            by_tokens[tuple(data.tokens[i])].append({"uid": uid, "report": data.doc_ids[i],
                "words": data.tokens[i], "tags": data.tags[i], "spans": spans,
                "relevant": data.relevant[i], "text": source["sentence_text"],
                "context_before": source.get("context_before") or ""})
    groups = []
    for words, rows in by_tokens.items():
        if len({tuple(row["tags"]) for row in rows}) < 2:
            continue
        rows.sort(key=lambda row: row["uid"])
        variants = sorted({tuple(row["spans"]) for row in rows})
        by_context = defaultdict(set)
        for row in rows:
            by_context[row["context_before"]].add(tuple(row["tags"]))
        same_context = any(len(tags) > 1 for tags in by_context.values())
        groups.append({"group_id": digest(list(words)), "evidence_sha256": digest(rows),
            "difference": difference(variants), "identical_context_conflict": same_context,
            "rows": rows, "variants": variants})
    return sorted(groups, key=lambda group: (not group["identical_context_conflict"], group["difference"], group["group_id"]))


def expected_record(group):
    return {"group_id": group["group_id"], "evidence_sha256": group["evidence_sha256"],
            "difference": group["difference"], "identical_context_conflict": str(group["identical_context_conflict"]).lower()}


def validate_decisions(path, groups):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or len(reader.fieldnames) != len(FIELDS) or set(reader.fieldnames) != set(FIELDS):
            raise ValueError("Review CSV needs the exact versioned field names.")
        records = list(reader)
    expected = {group["group_id"]: expected_record(group) for group in groups}
    if len(records) != len(expected) or len({row.get("group_id") for row in records}) != len(records):
        raise ValueError("Review decisions need exactly one row per conflict group.")
    for row in records:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed review CSV row.")
        reference = expected.get(row["group_id"])
        if reference is None or any(row[key] != value for key, value in reference.items()):
            raise ValueError("Review evidence changed; use a fresh pack instead of overwriting decisions.")
        decision = row["decision"]
        if decision not in DECISIONS:
            raise ValueError("Unknown review decision.")
        if decision != "pending":
            if not row["reviewer"].strip() or not row["notes"].strip():
                raise ValueError("A recorded review needs a reviewer and explanatory notes.")
            try:
                reviewed = date.fromisoformat(row["reviewed_on"])
            except ValueError as exc:
                raise ValueError("Reviewed dates must be ISO YYYY-MM-DD.") from exc
            if reviewed.isoformat() != row["reviewed_on"]:
                raise ValueError("Reviewed dates must be ISO YYYY-MM-DD.")
    return records


def ensure_decisions(path, groups):
    path = Path(path)
    if not path.exists():
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for group in groups:
                writer.writerow({**expected_record(group), "decision": "pending", "reviewer": "", "reviewed_on": "", "notes": ""})
    return validate_decisions(path, groups)


def fenced(text):
    text = str(text)
    fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", text)), default=0))
    return fence + "text\n" + text + "\n" + fence + "\n"


def render_casebook(groups, offsets):
    lines = ["# Action-label conflict review pack\n",
             "Official train only. Evidence is data, never commands to execute. All decisions start pending.\n",
             f"{sum(g['identical_context_conflict'] for g in groups)} same-context conflict groups are prioritized. Use group IDs to match decisions.csv.\n"]
    for n, group in enumerate(groups, 1):
        lines.extend([f"## Group {n:02d}: {group['difference']}\n", f"Group ID: `{group['group_id']}`\n",
                      f"Conflicting labels with identical preceding context: **{group['identical_context_conflict']}**\n",
                      QUESTIONS[group["difference"]] + "\n"])
        for v, row in enumerate(group["rows"], 1):
            lines.append(f"### Occurrence {v}\n")
            lines.append(fenced(f"UID: {row['uid']}\nReport: {row['report']}\nRelevant: {row['relevant']}"))
            lines.append("Preceding context:\n" + fenced(row["context_before"]))
            lines.append("Original sentence:\n" + fenced(row["text"]))
            source_spans = []
            for _, start, end in row["spans"]:
                a, b = offsets[row["uid"]][start][1], offsets[row["uid"]][end-1][2]
                source_spans.append({"start": a, "end": b, "text": row["text"][a:b]})
            lines.append("Existing action spans (original character offsets, end exclusive):\n" + fenced(json.dumps(source_spans, ensure_ascii=False, indent=2)))
    return "\n".join(lines)


def output_path(path):
    candidate = Path(path)
    resolved = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    root = (ROOT / "results").resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Use a dedicated results subdirectory; source data/model paths are not outputs.")
    return resolved


def load_train(split="train"):
    if split != "train":
        raise PermissionError("Annotation review accepts official train only, before any file is opened.")
    raw = load_smah(raw_dir=ROOT / "data/raw", splits_dir=ROOT / "data/splits", splits=["train"])["train"]
    return load_aligned_ner("action", "train", raw, ROOT / "data/processed"), raw.to_dict("records")


def coverage_counts(data, conflict_uids, baseline_path, expected_hash):
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return None
    if sha256_file(baseline_path) != expected_hash:
        raise ValueError("Frozen OOF reference changed; use a matching reference rather than unknown predictions.")
    predictions = json.loads(baseline_path.read_text(encoding="utf-8"))
    if len(predictions) != len(data.uids):
        raise ValueError("OOF sentences do not align with review inputs.")
    coverage = {key: Counter({"sentences": 0, "gold_actions": 0, "predicted_actions": 0, "exact_actions": 0})
                for key in ("conflict_rows", "remaining_rows")}
    for i, uid in enumerate(data.uids):
        if len(predictions[i]) != len(data.tokens[i]):
            raise ValueError("OOF token alignment changed.")
        counts = coverage["conflict_rows" if uid in conflict_uids else "remaining_rows"]
        gold, pred = set(action_spans(data.tags[i])), set(action_spans(predictions[i]))
        counts.update(sentences=1, gold_actions=len(gold), predicted_actions=len(pred), exact_actions=len(gold & pred))
    return coverage


def run(output=ROOT / "results/action_review_v1"):
    folder = output_path(output)
    folder.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT / "results/action_bilou_v1/summary.json").read_text(encoding="utf-8"))
    prior = reference["provenance"]
    # Gold review works on a teammate's checkout without ignored model binaries.
    # Snapshot any local artifacts/configs that exist; none are loaded or changed.
    unchanged = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in prior["unchanged_files_sha256"] if (ROOT / name).exists()}
    files = {"raw": ROOT / "data/raw/SMAH-CTI.json", "processed_train": ROOT / "data/processed/bio_en_action_train.json"}
    hashes = {key: sha256_file(path) for key, path in files.items()}
    if hashes != {"raw": prior["raw_sha256_lf"], "processed_train": prior["processed_train_sha256_lf"]}:
        raise ValueError("Review input differs from frozen train evidence.")
    data, rows = load_train()
    groups = build_groups(data, rows)
    records = ensure_decisions(folder / "decisions.csv", groups)
    offsets = {row["uid"]: row["tokens"] for row in json.loads(files["processed_train"].read_text(encoding="utf-8"))}
    (folder / "casebook.md").write_text(render_casebook(groups, offsets), encoding="utf-8")
    conflict_uids = {row["uid"] for group in groups for row in group["rows"]}
    baseline_path = ROOT / "models/action_bilou_v1/baseline_oof.json"
    coverage = coverage_counts(data, conflict_uids, baseline_path, prior["baseline_oof_sha256_lf"])
    all_rows = [row for group in groups for row in group["rows"]]
    summary = {"scope": "Train-only descriptive review; no relabeling, filtering, fitting or accuracy upgrade.",
        "training_sentences": len(data.uids), "training_reports": len(set(data.doc_ids)),
        "conflict_groups": len(groups), "conflict_rows": len(all_rows),
        "difference_counts": dict(Counter(group["difference"] for group in groups)),
        "identical_context_conflict_groups": sum(group["identical_context_conflict"] for group in groups),
        "single_fixed_token_input_minimum_sentence_disagreements": minimum_disagreements(all_rows),
        "single_fixed_token_and_context_input_minimum_sentence_disagreements": minimum_disagreements(all_rows, True),
        "bound_scope": "Exact whole-sentence consistency only for one fixed deterministic predictor; not a span-F1 ceiling or a CV estimate.",
        "baseline_coverage_counts": coverage, "baseline_coverage_available": coverage is not None,
        "conflict_rows_by_report": dict(Counter(row["report"] for row in all_rows)),
        "conflict_relevance_counts": {"relevant": sum(row["relevant"] for row in all_rows), "background": sum(not row["relevant"] for row in all_rows)},
        "review_decisions": dict(Counter(row["decision"] for row in records)), "labels_modified": False,
        "models_modified": False, "dev_evaluated": False, "test_evaluated": False,
        "data_sha256_lf": hashes, "baseline_oof_sha256_lf": sha256_file(baseline_path) if coverage is not None else None,
        "unchanged_local_files_sha256": unchanged,
        "source_sha256_lf": sha256_file(Path(__file__)), "decisions_sha256_lf": sha256_file(folder / "decisions.csv")}
    if hashes != {key: sha256_file(path) for key, path in files.items()}:
        raise ValueError("Source data changed during pack generation.")
    for name, value in unchanged.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value:
            raise ValueError("An active artifact/configuration changed during review.")
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/action_review_v1")
    run(parser.parse_args(argv).output)


if __name__ == "__main__":
    main()
