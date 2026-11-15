import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from cti.core.dataset import load_smah
from cti.experiment_action import ROOT, choose_candidate, read_config, run_experiment
from cti.preprocessing.tokenizer import tokenize


def metric(value):
    return {"exact_strict": {"f1": value}}


def candidate_report(identifier, pooled, folds):
    return {"id": identifier, "pooled": metric(pooled), "fold_metrics": [metric(value) for value in folds]}


class CandidateSelectionTests(unittest.TestCase):
    def test_requires_pooled_gain_and_multiple_fold_wins(self):
        baseline = candidate_report("baseline", .20, [.20, .20, .20])
        single_fold = candidate_report("one_fold", .25, [.40, .19, .19])
        too_small = candidate_report("small_gain", .202, [.21, .21, .21])
        self.assertEqual(choose_candidate([baseline, single_fold, too_small], .005, 2), "baseline")
        consistent = candidate_report("consistent", .23, [.22, .24, .23])
        self.assertEqual(choose_candidate([baseline, single_fold, consistent], .005, 2), "consistent")

    def test_ties_retain_baseline_or_earlier_preregistered_candidate(self):
        baseline = candidate_report("baseline", .20, [.20] * 3)
        tied = candidate_report("tied", .20, [.20] * 3)
        self.assertEqual(choose_candidate([baseline, tied], .005, 2), "baseline")
        first = candidate_report("first", .23, [.23] * 3)
        second = candidate_report("second", .23, [.23] * 3)
        self.assertEqual(choose_candidate([baseline, first, second], .005, 2), "first")


class ActionExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for directory in ("raw", "splits", "processed"):
            (self.root / directory).mkdir()
        raw = []
        self.split_ids = {}
        for split, count in (("train", 6), ("dev", 2), ("test", 1)):
            identifiers, records = [], []
            for report in range(count):
                doc_id = f"synthetic-{split}-{report}"
                identifiers.append(doc_id)
                for index, (text, relevant) in enumerate([("APT29 used Mimikatz.", "yes"), ("Background information.", "no")]):
                    uid = f"{doc_id}-{index}"
                    raw.append({"uid": uid, "doc_id": doc_id, "sentence_text": text, "relevant": relevant, "source": "synthetic"})
                    pieces = tokenize(text)
                    tags = ["O", "B-ACTION", "I-ACTION", "O"] if relevant == "yes" else ["O"] * len(pieces)
                    records.append({"uid": uid, "tokens": [list(piece) + [tag] for piece, tag in zip(pieces, tags)]})
            self.split_ids[split] = identifiers
            (self.root / f"splits/{split}_report_ids.json").write_text(json.dumps(identifiers), encoding="utf-8")
            if split != "test":
                (self.root / f"processed/bio_en_action_{split}.json").write_text(json.dumps(records), encoding="utf-8")
        (self.root / "raw/SMAH-CTI.json").write_text(json.dumps(raw), encoding="utf-8")
        params = {"c1": 0, "c2": .01, "max_iterations": 15}
        self.config = {
            "folds": 3, "seed": 42, "minimum_cv_gain": .005, "minimum_fold_wins": 2,
            "candidates": [
                {"id": "baseline", "parameters": params},
                {"id": "comparison", "parameters": copy.deepcopy(params)},
            ],
        }
        self.path = self.root / "config.yaml"
        self.write_config()

    def write_config(self):
        self.path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def run_trial(self, model_dir=None):
        return run_experiment(
            self.path, self.root / "raw", self.root / "splits", self.root / "processed",
            self.root / "models" if model_dir is None else model_dir, self.root / "results",
        )

    def test_config_rejects_arbitrary_models_and_nonlocal_feature_inputs(self):
        original = copy.deepcopy(self.config)
        for change in (
            lambda cfg: cfg["candidates"][0].update(model="bert_ner"),
            lambda cfg: cfg["candidates"][0]["parameters"].update(epochs=5),
            lambda cfg: cfg["candidates"][0]["parameters"].update(feature_config={"word_clusters": {"dev": 1}}),
            lambda cfg: cfg["candidates"][0]["parameters"].update(feature_config={"use_gazetteer": "false"}),
            lambda cfg: cfg["candidates"].append(copy.deepcopy(cfg["candidates"][0])),
        ):
            self.config = copy.deepcopy(original)
            change(self.config)
            self.write_config()
            with self.assertRaises(ValueError):
                read_config(self.path)

    def test_selection_is_frozen_before_dev_and_only_train_reports_enter_cv(self):
        calls = []

        def checked_load(**kwargs):
            calls.append(kwargs["splits"])
            if kwargs["splits"] == ["dev"]:
                selection = json.loads((self.root / "results/selection.json").read_text())
                self.assertEqual(selection["selected"], "baseline")
                self.assertFalse(selection["dev_labels_used_for_selection"])
            return load_smah(**kwargs)

        with patch.dict(os.environ, {"CTI_ALLOW_TEST": "1"}), patch("cti.experiment_action.load_smah", side_effect=checked_load):
            summary = self.run_trial()
        self.assertEqual(calls, [["train"], ["dev"]])
        self.assertFalse(summary["test_split_used"])
        self.assertFalse(summary["active_models_modified"])
        cv = json.loads((self.root / "results/cv.json").read_text())
        held_out = []
        for fold in cv["folds"]:
            fit, validation = set(fold["train_report_ids"]), set(fold["validation_report_ids"])
            self.assertFalse(fit & validation)
            self.assertEqual(fit | validation, set(self.split_ids["train"]))
            held_out.extend(validation)
        self.assertEqual(sorted(held_out), sorted(self.split_ids["train"]))
        self.assertTrue((self.root / "models/baseline_action.joblib").is_file())
        report = json.loads((self.root / "results/baseline_dev.json").read_text())
        self.assertTrue(report["reload_predictions_identical"])
        self.assertEqual(report["metrics_version"], "2")

    def test_matching_fold_caches_resume_without_refitting_cv(self):
        self.run_trial()
        from cti.experiment_action import _fit
        sizes = []

        def checked_fit(candidate, data, indices):
            indices = list(indices)
            sizes.append(len(indices))
            return _fit(candidate, data, indices)

        with patch("cti.experiment_action._fit", side_effect=checked_fit):
            self.run_trial()
        self.assertEqual(sizes, [12])  # Only the final full-train confirmation.

    def test_corrupt_cached_predictions_are_rejected(self):
        self.run_trial()
        cache = self.root / "models/baseline_fold_1.json"
        content = json.loads(cache.read_text())
        content["predictions"][0][0] = "B-ACTION"
        cache.write_text(json.dumps(content), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            self.run_trial()

    def test_protected_or_overlapping_split_ids_fail_before_loading_rows(self):
        path = self.root / "splits/test_report_ids.json"
        path.write_text(json.dumps([self.split_ids["train"][0]]), encoding="utf-8")
        with patch("cti.experiment_action.load_smah") as loader:
            with self.assertRaises(PermissionError):
                self.run_trial()
        loader.assert_not_called()

    def test_active_model_directory_cannot_be_used_for_experiment_output(self):
        with self.assertRaisesRegex(ValueError, "separate experiment"):
            self.run_trial(model_dir=ROOT / "models")


if __name__ == "__main__":
    unittest.main()
