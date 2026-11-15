"""Synthetic fixtures only; no protected examples, labels or real user data."""

import csv
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cti.sms_intake import (FIELDS, audit_annotations, freeze_batch, group_campaigns,
                            load_intake, local_output, main, protected_references, read_annotations, resolve_annotation, sha)


def annotation(identity="a", text="Bring milk and bread to the family dinner tomorrow.", label="ham", **changes):
    # Invented fixture content exercises the real-origin gate. These records
    # exist only inside tests and are never added to a project dataset.
    row = dict.fromkeys(FIELDS, "")
    row.update(id=identity, text=text, source_id="fixture-source", source_reference="fixture://test-only",
               usage_permission="consented_research", observed_date="2026-11-15", language="en", origin="real",
               category="personal", reviewer_a="alice", label_a=label, reviewer_b="bob", label_b=label,
               rationale="Synthetic fixture for testing annotation validation.")
    return {**row, **changes}


def independent_fixtures():
    rng = random.Random(89)
    rows = []
    for i in range(30):
        # Random letter strings deliberately avoid numeric-template collapse
        # and accidental similarity; they are not linguistic training examples.
        text = " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=12)) for _ in range(6))
        rows.append(annotation(f"fixture-{i:02d}", text, "ham" if i % 2 else "spam"))
    return audit_annotations(rows, [])[0]


class SMSIntakeTests(unittest.TestCase):
    def test_csv_preserves_source_unicode_newlines_and_punctuation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "annotations.csv"
            text = "\ufeff\U0001f6e1 A message, with quoted \"words\".\r\nSecond line."
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader(); writer.writerow(annotation(text=text))
            rows, digest = read_annotations(path)
            self.assertEqual(rows[0]["text"], text)
            self.assertEqual(len(digest), 64)

    def test_bad_headers_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "annotations.csv"
            path.write_text("id,text,id\na,b,c\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "columns"):
                read_annotations(path)
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader(); writer.writerows([annotation(), annotation()])
            with self.assertRaisesRegex(ValueError, "unique"):
                read_annotations(path)

    def test_consensus_requires_two_distinct_reviewers(self):
        self.assertEqual(resolve_annotation(annotation()), ("ham", None))
        self.assertEqual(resolve_annotation(annotation(reviewer_b=" ALICE "))[1], "two_distinct_reviewers_required")

    def test_disagreement_requires_an_independent_explicit_adjudication(self):
        row = annotation(label_b="spam")
        self.assertEqual(resolve_annotation(row)[1], "independent_adjudication_required")
        row.update(adjudicator="alice", adjudicated_label="ham")
        self.assertEqual(resolve_annotation(row)[1], "independent_adjudication_required")
        row.update(adjudicator="carol")
        self.assertEqual(resolve_annotation(row), ("ham", None))

    def test_uncertain_and_synthetic_examples_do_not_enter_training(self):
        rows = [annotation("uncertain", label="uncertain"), annotation("generated", origin="synthetic")]
        accepted, quarantine, summary = audit_annotations(rows, [])
        self.assertEqual(accepted, [])
        self.assertEqual({r["reason"] for r in quarantine}, {"uncertain_label", "non_real_origin"})
        self.assertFalse(summary["model_used"])

    def test_metadata_and_scope_failures_are_quarantined(self):
        for change, expected in [({"source_reference": ""}, "missing_source_or_permission"),
                                 ({"usage_permission": "unknown"}, "missing_source_or_permission"),
                                 ({"observed_date": "2026-02-30"}, "invalid_observed_date"),
                                 ({"observed_date": "20261115"}, "invalid_observed_date"),
                                 ({"category": ""}, "missing_category_or_rationale"),
                                 ({"language": "pt"}, "unsupported_language"),
                                 ({"text": "x" * 1001}, "invalid_or_out_of_scope_text")]:
            with self.subTest(change=change):
                self.assertEqual(resolve_annotation(annotation(**change))[1], expected)

    def test_protected_canonical_and_near_matches_are_excluded(self):
        text = "Your banking account has been suspended. Please verify your details immediately."
        rows = [annotation("a", text), annotation("b", text.replace("account", "acount")),
                annotation("c", "Read the library guide before the hiking class next Tuesday.")]
        accepted, quarantine, _ = audit_annotations(rows, [text])
        self.assertEqual([r["id"] for r in accepted], ["c"])
        self.assertEqual({r["reason"] for r in quarantine}, {"protected_corpus_or_diagnostic_overlap"})

    def test_conflicting_template_labels_quarantine_every_version(self):
        rows = [annotation("a", "Receive 1234 tokens from the next round", "ham"),
                annotation("b", "Receive 5678 tokens from the next round", "spam")]
        accepted, quarantine, _ = audit_annotations(rows, [])
        self.assertEqual(accepted, [])
        self.assertEqual([r["reason"] for r in quarantine], ["conflicting_template_labels"] * 2)

    def test_protected_member_excludes_the_entire_connected_campaign(self):
        text = "A message from an old campaign reserved outside this training batch."
        rows = [annotation("a", text, campaign_id="same-campaign"),
                annotation("b", "The distant jungle bridge closed during the tropical storm", campaign_id="same-campaign"),
                annotation("c", "Buy a carton of milk for the school cafeteria")]
        accepted, quarantine, _ = audit_annotations(rows, [text])
        self.assertEqual([r["id"] for r in accepted], ["c"])
        self.assertEqual(quarantine, [{"id": "a", "reason": "protected_corpus_or_diagnostic_overlap"},
                                      {"id": "b", "reason": "protected_campaign_or_template_group_overlap"}])

    def test_duplicates_are_not_counted_as_independent_examples(self):
        rows = [annotation("b", "Collect 1234 boxes from the warehouse"),
                annotation("a", "Collect 5678 boxes from the warehouse")]
        accepted, quarantine, summary = audit_annotations(rows, [])
        self.assertEqual([r["id"] for r in accepted], ["a"])
        self.assertEqual(quarantine, [{"id": "b", "reason": "duplicate_template"}])
        self.assertEqual(summary["groups"], 1)
        self.assertEqual(audit_annotations(list(reversed(rows)), []), (accepted, quarantine, summary))

    def test_duplicate_alias_preserves_transitive_campaign_links(self):
        rows = [annotation("a", "Pick up 1234 crates from the office", campaign_id="campaign-one"),
                annotation("b", "Pick up 5678 crates from the office", campaign_id="campaign-two"),
                annotation("c", "The laboratory examination has moved to the chemistry building", campaign_id="campaign-one"),
                annotation("d", "Please bring fruit and milk for supper at the park", campaign_id="campaign-two")]
        accepted, quarantine, _ = audit_annotations(rows, [])
        self.assertEqual(len(accepted), 3)
        self.assertEqual(quarantine, [{"id": "b", "reason": "duplicate_template"}])
        self.assertEqual(len({r["group"] for r in accepted}), 1)

    def test_near_templates_share_a_group_without_a_declared_campaign(self):
        text = "Your banking account has been suspended. Please verify your details immediately."
        rows = group_campaigns([annotation("a", text), annotation("b", text.replace("account", "acount"))])
        self.assertEqual(rows[0]["group"], rows[1]["group"])

    def test_grouped_freeze_round_trip_and_repeatability(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = independent_fixtures()
            manifest = freeze_batch(rows, folder, {"input_sha256": "synthetic-only"})
            self.assertEqual(manifest, freeze_batch(rows, folder, {"input_sha256": "synthetic-only"}))
            train, dev = load_intake("train", folder), load_intake("dev", folder)
            self.assertEqual({r["id"] for r in train}, set(manifest["partitions"]["train"]))
            self.assertEqual({r["id"] for r in dev}, set(manifest["partitions"]["dev"]))
            self.assertTrue({r["group"] for r in train}.isdisjoint(r["group"] for r in dev))
            self.assertEqual(set(manifest["partitions"]), {"train", "dev", "test"})
            self.assertFalse(manifest["test_evaluated"])

    def test_too_few_groups_cannot_be_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = audit_annotations([annotation()], [])[0]
            with self.assertRaisesRegex(ValueError, "five independent"):
                freeze_batch(rows, folder, {})
            self.assertFalse((Path(folder) / "manifest.json").exists())

    def test_frozen_batch_cannot_be_overwritten_or_resplit(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = independent_fixtures()
            freeze_batch(rows, folder, {})
            original = (Path(folder) / "accepted.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "never overwrite"):
                freeze_batch(rows, folder, {"input_sha256": "changed"})
            self.assertEqual((Path(folder) / "accepted.jsonl").read_bytes(), original)

    def test_changed_snapshot_and_partition_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = independent_fixtures()
            manifest = freeze_batch(rows, folder, {})
            path = Path(folder) / "accepted.jsonl"
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                load_intake("train", folder)
            path.write_bytes(original)
            manifest["partitions"]["dev"].append(manifest["partitions"]["train"][0])
            (Path(folder) / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "IDs overlap"):
                load_intake("dev", folder)

    def test_external_canonical_protocol_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            freeze_batch(independent_fixtures(), folder, {})
            with patch("cti.sms_intake.protocol_sources", return_value={"changed": "digest"}):
                with self.assertRaisesRegex(ValueError, "protocol changed"):
                    load_intake("train", folder)

    def test_prior_batch_is_protected_and_own_batch_is_not_self_excluded(self):
        # All source files and hashes here are artificial; do not load the real
        # official holdouts even to test exclusion/access authorization.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            local = root / "local"
            previous = local / "previous"
            previous.mkdir(parents=True)
            examples = root / "examples"
            examples.mkdir()
            for name in ("message_challenge.json", "sms_collection_exclusions.json"):
                (examples / name).write_text(json.dumps({"messages": [{"text": name}]}), encoding="utf-8")
            uci, supplement = root / "uci", root / "supplement"
            uci.write_bytes(b"fake protected source")
            supplement.write_bytes(b"fake supplemental source")
            payload = b'{"text":"earlier reserved campaign message"}\n'
            snapshot = previous / "accepted.jsonl"
            snapshot.write_bytes(payload)
            (previous / "manifest.json").write_text(json.dumps({"schema_version": 1, "accepted_sha256": sha(payload)}), encoding="utf-8")
            with patch.multiple("cti.sms_intake", ROOT=root, LOCAL=local, DATA=uci, SOURCE=supplement,
                                DATA_SHA256=sha(uci.read_bytes())), \
                    patch("cti.sms_intake.read_rows", return_value=[{"text": "fake UCI message"}]), \
                    patch("cti.sms_intake.source_rows", return_value=[{"text": "fake supplement message"}]):
                texts, hashes = protected_references()
                self.assertIn("earlier reserved campaign message", texts)
                self.assertIn("local/previous/accepted.jsonl", hashes)
                texts, _ = protected_references(exclude_folder=previous)
                self.assertNotIn("earlier reserved campaign message", texts)
                snapshot.write_bytes(payload + b" ")
                with self.assertRaisesRegex(ValueError, "snapshot has changed"):
                    protected_references()

    def test_cli_freeze_is_idempotent_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_path, output = root / "annotations.csv", root / "batch"
            with input_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader(); writer.writerows(independent_fixtures())
            args = ["check", str(input_path), "--output-dir", str(output), "--freeze"]
            with patch("cti.sms_intake.LOCAL", root), patch("sys.stdout", new_callable=io.StringIO):
                with patch("cti.sms_intake.protected_references", return_value=([], {})):
                    main(args)
                before = {p.name: p.read_bytes() for p in output.iterdir()}
                with patch("cti.sms_intake.protected_references", side_effect=AssertionError("Must not rebuild frozen references")):
                    main(args)
                self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
                input_path.write_bytes(input_path.read_bytes() + b"\n")
                with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_test_lock_rejects_before_any_file_access(self):
        with patch("pathlib.Path.read_text", side_effect=AssertionError("no reads")), patch("pathlib.Path.read_bytes", side_effect=AssertionError("no reads")):
            with self.assertRaisesRegex(ValueError, "reserved"):
                load_intake("test", "does-not-exist")

    def test_cli_cannot_write_raw_annotations_outside_ignored_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "ignored"):
                local_output(Path(folder) / "public.csv")
            with patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    main(["init", "--output", str(Path(folder) / "public.csv")])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((Path(folder) / "public.csv").exists())

    def test_quarantine_summary_does_not_export_message_text(self):
        private = "Some sensitive family conversation which must never appear in a summary."
        _, quarantine, summary = audit_annotations([annotation(text=private, label_b="spam")], [])
        self.assertNotIn(private, json.dumps(quarantine))
        self.assertNotIn(private, json.dumps(summary))


if __name__ == "__main__":
    unittest.main()
