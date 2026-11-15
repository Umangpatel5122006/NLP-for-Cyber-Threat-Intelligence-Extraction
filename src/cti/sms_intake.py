"""Review local SMS annotations and freeze new grouped train/dev/test batches.

This module does not train, predict, fetch URLs, or modify existing corpora.
"""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import date
from pathlib import Path

from cti.core.sms_data import DATA, DATA_SHA256, ROOT, read_rows, validate_split
from cti.core.sms_domain import SOURCE, audit_overlap, canonical, source_rows


LOCAL = ROOT / "data/local_sms"
FIELDS = ("id", "text", "source_id", "source_reference", "usage_permission", "observed_date",
          "language", "origin", "category", "campaign_id", "reviewer_a", "label_a", "reviewer_b",
          "label_b", "adjudicator", "adjudicated_label", "rationale")
CATEGORIES = ("financial_notification", "delivery", "employment", "investment", "grant_or_prize",
              "credential_request", "warning", "personal", "other")
LABELS = ("ham", "spam", "uncertain")
POLICY = "sms_intake_v1; char_wb(3,5) hashing cosine>=0.85; campaign union; 5 group folds seed91"
MAX_ROWS = 2000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def protocol_sources():
    names = ("src/cti/sms_intake.py", "src/cti/core/sms_domain.py", "src/cti/core/sms_data.py")
    return {name: sha((ROOT / name).read_bytes().replace(b"\r\n", b"\n")) for name in names}


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def read_annotations(path):
    raw = Path(path).read_bytes()
    if len(raw) > 8_000_000:
        raise ValueError("Batch exceeds 8 MB; use separate reviewed batches.")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != set(FIELDS):
        raise ValueError("CSV columns must match examples/sms_annotation_template.csv exactly.")
    rows = list(reader)
    if not rows or len(rows) > MAX_ROWS:
        raise ValueError(f"Supply 1 to {MAX_ROWS} rows per batch.")
    ids = set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed CSV record; quote text containing commas or newlines.")
        for key in FIELDS:
            if key != "text":
                row[key] = row[key].strip()
        if not row["id"] or row["id"] in ids:
            raise ValueError("Each row needs a unique nonempty ID.")
        ids.add(row["id"])
    return rows, sha(raw)


def resolve_annotation(row):
    """Return a reviewed label or a quarantine reason, never a model-derived label."""
    if not row["text"].strip() or "\0" in row["text"] or len(row["text"]) > 1000 or not canonical(row["text"]):
        return None, "invalid_or_out_of_scope_text"
    if row["origin"] != "real":
        return None, "non_real_origin"
    if row["language"] != "en":
        return None, "unsupported_language"
    if not row["source_id"] or not row["source_reference"] or row["usage_permission"] not in ("public_dataset", "consented_research"):
        return None, "missing_source_or_permission"
    try:
        parsed = date.fromisoformat(row["observed_date"])
        if parsed.isoformat() != row["observed_date"]:
            raise ValueError("Use an ISO calendar date.")
    except ValueError:
        return None, "invalid_observed_date"
    if row["category"] not in CATEGORIES or not row["rationale"]:
        return None, "missing_category_or_rationale"
    a, b = row["reviewer_a"].strip().casefold(), row["reviewer_b"].strip().casefold()
    if not a or not b or a == b:
        return None, "two_distinct_reviewers_required"
    left, right, final = row["label_a"], row["label_b"], row["adjudicated_label"]
    if left not in LABELS or right not in LABELS or final not in (*LABELS, ""):
        return None, "invalid_label"
    if left == right and (not final or final == left):
        label = left
    else:
        reviewer = row["adjudicator"].strip().casefold()
        if not reviewer or reviewer in (a, b) or not final:
            return None, "independent_adjudication_required"
        label = final
    return (label, None) if label in ("ham", "spam") else (None, "uncertain_label")


def protected_references(exclude_folder=None):
    """Text-only exclusions across whole old corpora; no evaluation or fitting."""
    if sha(DATA.read_bytes()) != DATA_SHA256:
        raise ValueError("Protected UCI corpus differs from the pinned source.")
    paths = [DATA, SOURCE, ROOT / "examples/message_challenge.json", ROOT / "examples/sms_collection_exclusions.json"]
    texts = [r["text"] for r in read_rows()] + [r["text"] for r in source_rows()]
    for path in paths[2:]:
        texts.extend(r["text"] for r in json.loads(path.read_text(encoding="utf-8"))["messages"])
    hashes = {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in paths}
    # Earlier private batches include their own reserved rows. Use text only;
    # never infer or use protected labels to rank incoming messages.
    for definition in sorted(LOCAL.rglob("manifest.json")):
        if exclude_folder is not None and definition.parent.resolve() == Path(exclude_folder).resolve():
            continue
        snapshot = definition.with_name("accepted.jsonl")
        manifest = json.loads(definition.read_text(encoding="utf-8"))
        payload = snapshot.read_bytes()
        if manifest.get("schema_version") != 1 or sha(payload) != manifest.get("accepted_sha256"):
            raise ValueError("An earlier protected intake snapshot has changed.")
        texts.extend(json.loads(line)["text"] for line in payload.decode("utf-8").splitlines())
        hashes[snapshot.relative_to(ROOT).as_posix()] = sha(payload)
    return texts, hashes


def group_campaigns(rows):
    """Connect near templates and declared campaigns before dropping aliases."""
    from sklearn.feature_extraction.text import HashingVectorizer

    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def join(i, j):
        a, b = find(i), find(j)
        parents[max(a, b)] = min(a, b)
    campaigns, exact = {}, {}
    for i, row in enumerate(rows):
        key = canonical(row["text"])
        join(i, exact.setdefault(key, i))
        if row["campaign_id"]:
            join(i, campaigns.setdefault(row["campaign_id"], i))
    if rows:
        vectorizer = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**18,
                                      alternate_sign=False, norm="l2")
        matrix = vectorizer.transform([canonical(r["text"]) for r in rows])
        for start in range(0, len(rows), 128):
            edges = (matrix[start:start + 128] @ matrix.T).tocoo()
            for offset, j, value in zip(edges.row, edges.col, edges.data):
                i = start + offset
                if i < j and value >= 0.85:
                    join(i, j)
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(find(i), []).append(row["id"])
    return [{**row, "group": "intake-" + sha("|".join(sorted(members[find(i)])).encode())}
            for i, row in enumerate(rows)]


def audit_annotations(rows, protected_texts):
    accepted, quarantined = [], []
    def exclude(row, reason):
        quarantined.append({"id": row["id"], "reason": reason})
    for row in sorted(rows, key=lambda r: r["id"]):
        label, reason = resolve_annotation(row)
        if reason:
            exclude(row, reason)
        else:
            accepted.append({**row, "label": label, "original_label": label})
    labels = {}
    for row in accepted:
        labels.setdefault(canonical(row["text"]), set()).add(row["label"])
    # Group the complete reviewed batch first: a removed alias must not break
    # a campaign bridge and let its surviving members cross partitions.
    accepted = group_campaigns(accepted)
    conflicts = {key for key, values in labels.items() if len(values) > 1}
    if accepted and protected_texts:
        remaining, _ = audit_overlap(accepted, protected_texts)
        retained = {r["id"] for r in remaining}
    else:
        retained = {r["id"] for r in accepted}
    protected_groups = {r["group"] for r in accepted if r["id"] not in retained}
    unique, aliases = [], {}
    for row in accepted:
        if canonical(row["text"]) in conflicts:
            exclude(row, "conflicting_template_labels")
        elif row["id"] not in retained:
            exclude(row, "protected_corpus_or_diagnostic_overlap")
        elif row["group"] in protected_groups:
            exclude(row, "protected_campaign_or_template_group_overlap")
        else:
            key = canonical(row["text"])
            if key in aliases:
                exclude(row, "duplicate_template")
            else:
                aliases[key] = row["id"]
                unique.append(row)
    source_labels = {source: dict(Counter(r["label"] for r in unique if r["source_id"] == source))
                     for source in sorted({r["source_id"] for r in unique})}
    summary = {"input_rows": len(rows), "accepted_rows": len(unique), "quarantined_rows": len(quarantined),
               "quarantine_reasons": dict(sorted(Counter(r["reason"] for r in quarantined).items())),
               "labels": dict(sorted(Counter(r["label"] for r in unique).items())),
               "source_labels": source_labels,
               "single_label_sources": [source for source, counts in source_labels.items() if len(counts) == 1],
               "category_labels": {category: dict(Counter(r["label"] for r in unique if r["category"] == category))
                                   for category in CATEGORIES},
               "groups": len({r["group"] for r in unique}), "model_used": False, "test_evaluated": False}
    return unique, quarantined, summary


def freeze_batch(rows, folder, provenance):
    from sklearn.model_selection import StratifiedGroupKFold

    # This minimum permits splitting; it is not a claim of adequate sample size.
    if any(len({r["group"] for r in rows if r["label"] == label}) < 5 for label in ("ham", "spam")):
        raise ValueError("Need at least five independent groups per label before freezing.")
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=91).split(
        rows, [r["label"] for r in rows], [r["group"] for r in rows]))
    test, dev = set(folds[0][1]), set(folds[1][1])
    partitions = {"train": [], "dev": [], "test": []}
    for i, row in enumerate(rows):
        partitions["test" if i in test else "dev" if i in dev else "train"].append(row["id"])
    validate_split(rows, {"partitions": partitions})
    for values in partitions.values():
        if {r["label"] for r in rows if r["id"] in values} != {"ham", "spam"}:
            raise ValueError("Grouped split cannot represent both labels in each partition; collect more groups.")
    payload = b"".join((json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n").encode() for r in rows)
    manifest = {"schema_version": 1, "policy": POLICY, "protocol_source_sha256": protocol_sources(),
                "provenance": provenance, "accepted_sha256": sha(payload), "partitions": partitions,
                "groups": {r["id"]: r["group"] for r in rows}, "test_evaluated": False}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot, definition = folder / "accepted.jsonl", folder / "manifest.json"
    if snapshot.exists() or definition.exists():
        if not snapshot.exists() or not definition.exists() or snapshot.read_bytes() != payload or definition.read_bytes() != json_bytes(manifest):
            raise ValueError("Frozen batch differs; use a new batch directory, never overwrite or resplit it.")
    else:
        snapshot.write_bytes(payload)
        definition.write_bytes(json_bytes(manifest))
    return manifest


def load_intake(split, folder):
    if split not in ("train", "dev"):
        raise ValueError("Intake test split is reserved for final frozen evaluation.")
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    payload = (folder / "accepted.jsonl").read_bytes()
    if (manifest.get("schema_version") != 1 or manifest.get("policy") != POLICY
            or manifest.get("protocol_source_sha256") != protocol_sources() or sha(payload) != manifest.get("accepted_sha256")):
        raise ValueError("Frozen intake data or protocol changed.")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    if len({r["id"] for r in rows}) != len(rows) or manifest["groups"] != {r["id"]: r["group"] for r in rows}:
        raise ValueError("Frozen intake row identities or groups changed.")
    if any(resolve_annotation(r) != (r["label"], None) for r in rows):
        raise ValueError("Frozen intake includes an unresolved annotation.")
    validate_split(rows, manifest)
    selected = set(manifest["partitions"][split])
    return [r for r in rows if r["id"] in selected]


def local_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(LOCAL.resolve()):
        raise ValueError("Keep intake outputs inside ignored data/local_sms/.")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create an empty private CSV template.")
    init.add_argument("--output", type=Path, default=LOCAL / "annotations.csv")
    check = sub.add_parser("check", help="Audit reviewed annotations; no model calls.")
    check.add_argument("input", type=Path)
    check.add_argument("--output-dir", type=Path, default=LOCAL / "batch_v1")
    check.add_argument("--freeze", action="store_true", help="Freeze accepted rows and grouped partitions; no training.")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            output = local_output(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(FIELDS)
            print("Empty annotation CSV created in ignored data/local_sms/.")
            return
        folder = local_output(args.output_dir)
        rows, input_sha = read_annotations(args.input)
        if (folder / "manifest.json").exists():
            definition = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if definition["provenance"]["input_sha256"] != input_sha:
                raise ValueError("Input differs from the frozen batch; use a new batch directory.")
            load_intake("train", folder)
            load_intake("dev", folder)
            for name, expected in definition["provenance"]["protected_source_sha256"].items():
                if sha((ROOT / name).read_bytes()) != expected:
                    raise ValueError("A frozen exclusion reference changed; do not silently rebuild this batch.")
            print(json.dumps({"already_frozen": True, "rows": {key: len(ids) for key, ids in definition["partitions"].items()}, "test_evaluated": False}))
            return
        references, hashes = protected_references(exclude_folder=folder)
        accepted, quarantine, summary = audit_annotations(rows, references)
        provenance = {"input_sha256": input_sha, "protected_source_sha256": hashes}
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "audit.json").write_bytes(json_bytes({**summary, "provenance": provenance}))
        (folder / "quarantine.json").write_bytes(json_bytes(quarantine))
        print(json.dumps({key: summary[key] for key in ("input_rows", "accepted_rows", "quarantined_rows", "groups", "labels")}, indent=2))
        if args.freeze:
            definition = freeze_batch(accepted, folder, provenance)
            print(json.dumps({"frozen_rows": {key: len(ids) for key, ids in definition["partitions"].items()}, "test_evaluated": False}))
    except (ValueError, OSError, UnicodeError, csv.Error) as error:
        # Per-row text is never included in errors or the console summary.
        parser.exit(2, f"Intake not completed: {error}\n")


if __name__ == "__main__":
    main()
