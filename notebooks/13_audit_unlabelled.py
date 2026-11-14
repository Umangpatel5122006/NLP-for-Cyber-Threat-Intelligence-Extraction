"""Audit unlabelled entity mentions appearing inside irrelevant sentences.

Gold entities exist only in relevant sentences (relevant == 'yes').
This script mines known entity strings from the training set's gold annotations,
scans irrelevant sentences (relevant == 'no') across train and dev,
computes match frequencies per label, and exports a 100-mention sample for human inspection.
"""

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "SMAH-CTI.json"
TRAIN_IDS_PATH = ROOT / "data" / "splits" / "train_report_ids.json"
DEV_IDS_PATH = ROOT / "data" / "splits" / "dev_report_ids.json"
OUTPUT_SAMPLE = ROOT / "results" / "unlabelled_mentions_sample.csv"


def is_valid_entity_string(text: str) -> bool:
    clean = text.strip().strip("'\"").strip()
    if len(clean) < 3:
        return False
    if clean.lower() in ENGLISH_STOP_WORDS:
        return False
    # Avoid overly long sentence-like clauses
    if len(clean.split()) > 5 or len(clean) > 50 or "\n" in clean:
        return False
    return True


def main():
    random.seed(42)

    with open(RAW_PATH, encoding="utf-8") as f:
        records = json.load(f)

    with open(TRAIN_IDS_PATH, encoding="utf-8") as f:
        train_doc_ids = set(json.load(f))

    with open(DEV_IDS_PATH, encoding="utf-8") as f:
        dev_doc_ids = set(json.load(f))

    # 1. Collect gold entities from training split
    train_records = [r for r in records if r.get("doc_id") in train_doc_ids]
    entity_counts_by_label = defaultdict(Counter)

    for r in train_records:
        if str(r.get("relevant")).casefold() == "yes":
            for e in r.get("entities") or []:
                lbl = e.get("label")
                txt = e.get("text", "").strip()
                if lbl and is_valid_entity_string(txt):
                    entity_counts_by_label[lbl][txt] += 1

    # Map each entity string to its most frequent gold label in train
    entity_to_label = {}
    term_patterns = {}

    for lbl, counter in entity_counts_by_label.items():
        for term, count in counter.items():
            if term not in entity_to_label or count > entity_to_label[term][1]:
                entity_to_label[term] = (lbl, count)

    print(f"Collected {len(entity_to_label)} distinct valid entity strings from train gold:")
    label_distribution = Counter(lbl for lbl, _ in entity_to_label.values())
    for lbl, c in sorted(label_distribution.items()):
        print(f"  {lbl}: {c} distinct terms")

    # Compile word-boundary regex patterns for matching
    compiled_patterns = []
    for term, (lbl, _) in entity_to_label.items():
        # Use word boundaries if term starts/ends with alphanumeric
        pattern_str = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            compiled_patterns.append((pattern, term, lbl))
        except re.error:
            continue

    # 2. Scan irrelevant sentences in train and dev splits
    target_doc_ids = train_doc_ids | dev_doc_ids
    irrelevant_records = [
        r for r in records
        if r.get("doc_id") in target_doc_ids and str(r.get("relevant")).casefold() != "yes"
    ]
    print(f"\nScanning {len(irrelevant_records)} irrelevant sentences across train and dev...")

    all_matches = []
    matches_per_label = Counter()

    for r in irrelevant_records:
        text = r.get("sentence_text", "")
        doc_id = r.get("doc_id")
        uid = r.get("uid")
        split = "train" if doc_id in train_doc_ids else "dev"

        # Check all patterns
        for pattern, gold_term, lbl in compiled_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                all_matches.append({
                    "split": split,
                    "doc_id": doc_id,
                    "uid": uid,
                    "label": lbl,
                    "entity_text": gold_term,
                    "matched_text": match.group(0),
                    "match_start": start,
                    "match_end": end,
                    "sentence_text": text,
                })
                matches_per_label[lbl] += 1

    print("\n=== Unlabelled Mention Audit Results ===")
    print(f"Total unlabelled entity matches in irrelevant sentences: {len(all_matches)}")
    for lbl, count in sorted(matches_per_label.items()):
        print(f"  {lbl}: {count} occurrences")

    # 3. Sample 100 occurrences for human evaluation
    sample_size = min(100, len(all_matches))
    sampled_matches = random.sample(all_matches, sample_size)

    OUTPUT_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split", "doc_id", "uid", "label", "entity_text",
        "matched_text", "match_start", "match_end", "sentence_text"
    ]
    with open(OUTPUT_SAMPLE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_matches)

    print(f"\nExported {len(sampled_matches)} sampled unlabelled mentions to: {OUTPUT_SAMPLE}")
    return matches_per_label, len(all_matches)


if __name__ == "__main__":
    main()
