"""Synthetic annotation-review tests; no official protected split is opened."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cti.action_review import (FIELDS, build_groups, coverage_counts, difference, ensure_decisions, fenced,
                               load_train, minimum_disagreements, output_path,
                               render_casebook, validate_decisions)


def fixtures():
    rows = [("a", ["run", "tool"], ["B-ACTION", "I-ACTION"], "same"),
            ("b", ["run", "tool"], ["B-ACTION", "O"], "same"),
            ("c", ["drop"], ["B-ACTION"], "before one"),
            ("d", ["drop"], ["O"], "before two")]
    data = SimpleNamespace(tokens=[r[1] for r in rows], tags=[r[2] for r in rows],
                           uids=[r[0] for r in rows], doc_ids=["report"] * 4, relevant=[True] * 4)
    raw = [{"uid": uid, "sentence_text": " ".join(words), "context_before": context} for uid, words, _, context in rows]
    return data, raw


class ActionReviewTests(unittest.TestCase):
    def test_grouping_prioritizes_identical_context_and_classifies_differences(self):
        data, raw = fixtures()
        groups = build_groups(data, raw)
        self.assertEqual(len(groups), 2)
        self.assertTrue(groups[0]["identical_context_conflict"])
        self.assertEqual(groups[0]["difference"], "boundary_extent")
        self.assertEqual(groups[1]["difference"], "presence_absence")

    def test_group_evidence_is_stable_under_raw_row_order(self):
        data, raw = fixtures()
        self.assertEqual(build_groups(data, raw), build_groups(data, list(reversed(raw))))

    def test_grouping_rejects_missing_or_duplicate_uid_metadata(self):
        data, raw = fixtures()
        for rows in (raw[:-1], [*raw, raw[0]]):
            with self.assertRaises(ValueError):
                build_groups(data, rows)

    def test_segmentation_and_disjoint_action_selection_are_distinct(self):
        self.assertEqual(difference([(("ACTION", 0, 5),), (("ACTION", 0, 2), ("ACTION", 3, 5))]), "span_count_or_segmentation")
        self.assertEqual(difference([(("ACTION", 0, 1),), (("ACTION", 2, 3),)]), "different_actions")

    def test_fixed_input_disagreement_bound_accounts_for_majority_and_context(self):
        rows = [{"words": ["x"], "tags": [tag], "context_before": context}
                for tag, context in (("O", "a"), ("O", "a"), ("B-ACTION", "b"))]
        self.assertEqual(minimum_disagreements(rows), 1)
        self.assertEqual(minimum_disagreements(rows, True), 0)
        rows[2]["context_before"] = "a"
        self.assertEqual(minimum_disagreements(rows, True), 1)

    def test_train_only_guard_rejects_other_splits_before_loader_call(self):
        with patch("cti.action_review.load_smah") as loader:
            for split in ("test", "dev", "all"):
                with self.assertRaises(PermissionError):
                    load_train(split)
            loader.assert_not_called()

    def test_review_does_not_require_a_local_model_prediction_cache(self):
        data, _ = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(coverage_counts(data, set(), Path(folder) / "missing.json", "digest"))

    def test_available_prediction_counts_are_hash_checked_and_conserve_rows(self):
        import json
        from cti.models.ner_artifact import sha256_file
        data, _ = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.json"
            path.write_text(json.dumps(data.tags), encoding="utf-8")
            counts = coverage_counts(data, {"a", "b"}, path, sha256_file(path))
            self.assertEqual(counts["conflict_rows"]["gold_actions"], 2)
            self.assertEqual(counts["remaining_rows"]["sentences"], 2)
            with self.assertRaisesRegex(ValueError, "reference changed"):
                coverage_counts(data, set(), path, "wrong hash")

    def test_initial_decisions_pending_and_regeneration_preserves_review_bytes(self):
        data, raw = fixtures()
        groups = build_groups(data, raw)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.csv"
            records = ensure_decisions(path, groups)
            self.assertTrue(all(r["decision"] == "pending" for r in records))
            records[0].update(decision="uncertain", reviewer="Reviewer A", reviewed_on="2026-11-16", notes="Need full report context.")
            self.write(path, records)
            before = path.read_bytes()
            returned = ensure_decisions(path, groups)
            self.assertEqual(returned[0]["decision"], "uncertain")
            self.assertEqual(path.read_bytes(), before)

    def test_reviewed_decisions_require_provenance_but_do_not_relabel(self):
        data, raw = fixtures()
        groups = build_groups(data, raw)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.csv"
            original = ensure_decisions(path, groups)
            for changes in ({"decision": "inconsistent_labels"},
                            {"decision": "uncertain", "reviewer": "A", "notes": "Reason", "reviewed_on": "bad-date"},
                            {"decision": "approved_for_training"}):
                rows = [dict(row) for row in original]
                rows[0].update(changes)
                self.write(path, rows)
                with self.assertRaises(ValueError):
                    validate_decisions(path, groups)
            self.assertEqual(data.tags[0], ["B-ACTION", "I-ACTION"])

    def test_evidence_changes_do_not_overwrite_existing_decisions(self):
        data, raw = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.csv"
            ensure_decisions(path, build_groups(data, raw))
            original = path.read_bytes()
            raw[0]["context_before"] = "changed evidence"
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                ensure_decisions(path, build_groups(data, raw))
            self.assertEqual(path.read_bytes(), original)

    def test_missing_duplicate_or_extra_groups_are_rejected(self):
        data, raw = fixtures()
        groups = build_groups(data, raw)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.csv"
            original = ensure_decisions(path, groups)
            for records in (original[:-1], [original[0], original[0]], [*original, original[0]]):
                self.write(path, records)
                with self.assertRaises(ValueError):
                    validate_decisions(path, groups)

    def test_output_must_be_a_results_subdirectory(self):
        with tempfile.TemporaryDirectory() as folder, patch("cti.action_review.ROOT", Path(folder)):
            self.assertEqual(output_path("results/review"), Path(folder) / "results/review")
            for path in ("data/raw", "results", "results/../../outside"):
                with self.assertRaises(ValueError):
                    output_path(path)

    def test_casebook_keeps_source_inside_fences_and_original_offsets(self):
        data, raw = fixtures()
        groups = build_groups(data, raw)
        offsets = {"a": [["run", 0, 3, "B-ACTION"], ["tool", 4, 8, "I-ACTION"]],
                   "b": [["run", 0, 3, "B-ACTION"], ["tool", 4, 8, "O"]],
                   "c": [["drop", 0, 4, "B-ACTION"]], "d": [["drop", 0, 4, "O"]]}
        book = render_casebook(groups, offsets)
        self.assertIn('"text": "run tool"', book)
        self.assertIn('"end": 8', book)
        self.assertTrue(fenced("```\ntext").startswith("````text\n"))

    @staticmethod
    def write(path, records):
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(records)


if __name__ == "__main__":
    unittest.main()
