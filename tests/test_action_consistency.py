"""Synthetic fit-only filtering checks; official validation labels stay fixed."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from cti.experiment_action_background import metrics
from cti.experiment_action_consistency import POLICIES, fit, read_config, run, select_fit_rows


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return SimpleNamespace(tokens=[["ambiguous"], ["ambiguous"], ["duplicate"], ["duplicate"], ["keep"], ["keep"], [], ["keep"]],
        tags=[["B-ACTION"], ["O"], ["B-ACTION"], ["B-ACTION"], ["B-ACTION"], ["O"], [], ["O"]],
        relevant=[True, True, True, True, True, False, True, True], uids=["u0", "u1", "z", "a", "u4", "u5", "u6", "u7"])


class ConsistencySelectionTests(unittest.TestCase):
    def test_ambiguous_inputs_are_removed_without_choosing_or_rewriting_a_label(self):
        data = fixture()
        original = copy.deepcopy(data.tags)
        selected, inventory = select_fit_rows(data, range(7), "consistent_inputs")
        self.assertEqual(selected, [2, 3, 4])
        self.assertEqual(inventory["ambiguous_rows_removed"], 2)
        self.assertEqual(inventory["duplicate_rows_removed"], 0)
        self.assertEqual(data.tags, original)

    def test_deduplication_uses_stable_uid_and_is_independent_of_fit_index_order(self):
        data = fixture()
        a, report = select_fit_rows(data, range(7), "unique_consistent_inputs")
        b, repeated = select_fit_rows(data, reversed(range(7)), "unique_consistent_inputs")
        self.assertEqual(a, [3, 4])
        self.assertEqual(a, b)
        self.assertEqual(report, repeated)
        self.assertEqual(report["duplicate_rows_removed"], 1)

    def test_held_out_tags_and_tokens_cannot_change_fit_selection(self):
        data = fixture()
        for policy in POLICIES:
            before = select_fit_rows(data, range(7), policy)
            changed = copy.deepcopy(data)
            changed.tokens[7] = ["ambiguous"]
            changed.tags[7] = ["B-ACTION"]
            self.assertEqual(select_fit_rows(changed, range(7), policy), before)

    def test_background_all_o_labels_do_not_make_a_relevant_input_ambiguous(self):
        data = fixture()
        for policy in POLICIES:
            selected, _ = select_fit_rows(data, [4, 5], policy)
            self.assertEqual(selected, [4])

    def test_empty_or_fully_ambiguous_fit_set_is_rejected(self):
        for indices in ([], [0, 1], [5, 6]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                select_fit_rows(fixture(), indices, "consistent_inputs")

    def test_invalid_indices_policies_and_duplicate_uids_are_rejected(self):
        for indices, policy in (([0, 0], "consistent_inputs"), ([-1], "consistent_inputs"), ([True], "consistent_inputs"), ([4], "unknown")):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                select_fit_rows(fixture(), indices, policy)
        data = fixture(); data.uids[1] = data.uids[0]
        with self.assertRaises(ValueError):
            select_fit_rows(data, [4], "consistent_inputs")

    def test_fit_receives_only_selected_examples(self):
        data = fixture()
        model = SimpleNamespace(fit=lambda tokens, tags: self.assertEqual(tokens, [["duplicate"], ["keep"]]))
        with patch("cti.experiment_action_consistency.create_crf_tagger", return_value=model):
            fitted, inventory = fit(data, range(7), "unique_consistent_inputs", {})
        self.assertIs(fitted, model)
        self.assertEqual(inventory["selected_sentences"], 2)

    def test_configuration_rejects_invalid_or_nonfinite_selection_budgets(self):
        original = yaml.safe_load((ROOT / "configs/action_consistency_experiment.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            for value in (True, float("nan"), float("inf"), -.1, 1.1):
                config = copy.deepcopy(original)
                config["selection"]["minimum_f1_gain"] = value
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_config(path)

    def test_rejected_experiment_never_loads_dev_or_predicts_with_full_active_model(self):
        config = yaml.safe_load((ROOT / "configs/action_consistency_experiment.yaml").read_text(encoding="utf-8"))
        data = SimpleNamespace(tokens=[[str(i)] for i in range(6)], tags=[["O"] for _ in range(6)], relevant=[True, False] * 3,
                               uids=[f"u{i}" for i in range(6)], doc_ids=["a", "a", "b", "b", "c", "c"])
        folds = [([i for i in range(6) if i not in (n, n+1)], [n, n+1]) for n in (0, 2, 4)]
        records = [{"train_report_ids": sorted({data.doc_ids[i] for i in a}), "validation_report_ids": sorted({data.doc_ids[i] for i in b}),
                    "validation_sentences": len(b)} for a, b in folds]
        baseline = [["O"] for _ in range(6)]
        class FakeCRF:
            def fit(self, tokens, tags):
                if len(tokens) != 2:
                    raise AssertionError("Only the two fit-relevant examples should be fitted.")
            def predict(self, tokens):
                return [["O"] for _ in tokens]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for directory in ("data/splits", "configs", "models/action_background_v1", "results/action_background_v1"):
                (root / directory).mkdir(parents=True)
            active = root / "models/ner_action.joblib"; active.write_bytes(b"synthetic active model")
            (root / "configs/config_ner.yaml").write_text("synthetic: true\n", encoding="utf-8")
            (root / "configs/action_background_experiment.yaml").write_text(yaml.safe_dump({"parameters": config["parameters"]}), encoding="utf-8")
            for name, ids in (("train", ["a", "b", "c"]), ("dev", ["d"]), ("test", ["t"])):
                (root / f"data/splits/{name}_report_ids.json").write_text(json.dumps(ids), encoding="utf-8")
            (root / "models/action_background_v1/relevant_only_oof.json").write_text(json.dumps(baseline), encoding="utf-8")
            prior = {"environment": {}, "source_sha256_lf": {}, "reference_cache_sha256_lf": {}, "unchanged_files_sha256": {
                     "models/ner_action.joblib": hashlib.sha256(active.read_bytes()).hexdigest()},
                     "raw_sha256_lf": "digest", "processed_train_sha256_lf": "digest", "config_sha256_lf": "digest",
                     "split_sha256_lf": {s: "digest" for s in ("train", "dev", "test")}}
            files = {"summary.json": {"selected": "relevant_only", "provenance": prior},
                     "integrity_audit.json": {"passed": True, "audit_script_sha256_lf": "digest", "oof_sha256_lf": {"models/action_background_v1/relevant_only_oof.json": "digest"}},
                     "selection.json": {"provenance": prior, "train_report_ids": ["a", "b", "c"], "candidates": [
                         {"id": "relevant_only", "pooled": metrics(data, baseline), "folds": [metrics(data, baseline, b) for _, b in folds]}]},
                     "cv.json": {"folds": records}}
            for name, value in files.items():
                (root / "results/action_background_v1" / name).write_text(json.dumps(value), encoding="utf-8")
            artifact = SimpleNamespace(manifest={"parameters": config["parameters"], "metadata": {"training_policy": "relevant_only"}})
            def train_only(**kwargs):
                self.assertEqual(kwargs["splits"], ["train"])
                return {"train": object()}
            with patch("cti.experiment_action_consistency.ROOT", root), \
                    patch("cti.experiment_action_consistency.read_config", return_value=config), \
                    patch("cti.experiment_action_consistency.load_smah", side_effect=train_only) as loader, \
                    patch("cti.experiment_action_consistency.load_aligned_ner", return_value=data), \
                    patch("cti.experiment_action_consistency.load_ner_tagger", return_value=artifact), \
                    patch("cti.experiment_action_consistency.get_report_grouped_splits", return_value=folds), \
                    patch("cti.experiment_action_consistency.sha256_file", return_value="digest"), \
                    patch("cti.experiment_action_consistency._environment", return_value={}), \
                    patch("cti.experiment_action_consistency.create_crf_tagger", side_effect=lambda **kwargs: FakeCRF()):
                result = run(root / "synthetic.yaml")
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(result["selected"], "baseline")
            self.assertFalse(result["dev_evaluated"])
            self.assertFalse(result["validation_filtered"])
            self.assertEqual(active.read_bytes(), b"synthetic active model")


if __name__ == "__main__":
    unittest.main()
