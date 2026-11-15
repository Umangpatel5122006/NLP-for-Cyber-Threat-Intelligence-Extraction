"""Synthetic sampling/selection/integration checks; no protected annotations."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from cti.experiment_action import _digest
from cti.experiment_action_background import POLICIES, checks, choose, read_config, run, training_indices


ROOT = Path(__file__).resolve().parents[1]


def synthetic():
    return SimpleNamespace(tokens=[[str(i)] for i in range(9)], tags=[["B-ACTION"]] * 3 + [["O"]] * 6,
                           relevant=[True] * 3 + [False] * 6, doc_ids=["a", "a", "b", "a", "a", "a", "b", "b", "c"],
                           uids=[f"u{i}" for i in range(9)])


class BackgroundPolicyTests(unittest.TestCase):
    def test_capped_sampling_is_per_report_and_deterministic_without_rewriting_gold(self):
        data = synthetic()
        original = copy.deepcopy(data.tags)
        selected = training_indices(data, range(9), "capped_background")
        self.assertEqual(selected, training_indices(data, reversed(range(9)), "capped_background"))
        self.assertTrue({0, 1, 2} <= set(selected))
        self.assertEqual(sum(not data.relevant[i] and data.doc_ids[i] == "a" for i in selected), 2)
        self.assertEqual(sum(not data.relevant[i] and data.doc_ids[i] == "b" for i in selected), 1)
        self.assertNotIn(8, selected)  # No relevant sentences in report c.
        self.assertEqual(data.tags, original)

    def test_fit_indices_exclude_held_out_reports_before_sampling(self):
        data = synthetic()
        fit = [i for i, report in enumerate(data.doc_ids) if report == "a"]
        for policy in POLICIES:
            selected = training_indices(data, fit, policy)
            self.assertTrue(set(selected) <= set(fit))
            self.assertEqual({data.doc_ids[i] for i in selected}, {"a"})

    def test_relevant_only_and_all_background_keep_expected_rows(self):
        data = synthetic()
        self.assertEqual(training_indices(data, range(9), "relevant_only"), [0, 1, 2])
        self.assertEqual(training_indices(data, range(9), "all_background"), list(range(9)))
        data.tokens[8] = []
        data.tags[8] = []
        self.assertEqual(training_indices(data, range(9), "all_background"), list(range(8)))

    def test_sampling_rejects_contradictory_background_and_invalid_indices(self):
        data = synthetic()
        for indices, policy in (([0, 0], "all_background"), ([-1], "all_background"), ([True], "all_background"),
                                ([3, 4], "capped_background"), ([0], "unknown")):
            with self.subTest(indices=indices, policy=policy), self.assertRaises(ValueError):
                training_indices(data, indices, policy)
        data.tags[3] = ["B-ACTION"]
        with self.assertRaisesRegex(ValueError, "all-O"):
            training_indices(data, range(9), "all_background")

    def test_policy_rejects_recall_loss_despite_better_precision_and_f1(self):
        policy = {"minimum_f1_gain": .005, "maximum_exact_recall_loss": .01, "minimum_background_reduction": .1}
        baseline = {"f1": .2, "recall": .3, "background_predicted_spans": 100}
        candidate = {"f1": .3, "recall": .28, "background_predicted_spans": 20}
        self.assertEqual(checks(baseline, candidate, policy), {"f1_gain": True, "exact_recall": False, "background_reduction": True})

    def test_selection_needs_gain_multiple_fold_wins_and_background_reduction(self):
        policy = {"minimum_f1_gain": .005, "maximum_exact_recall_loss": .01,
                  "minimum_background_reduction": .1, "minimum_fold_wins": 2}
        def report(name, value, background, folds):
            return {"id": name, "pooled": {"f1": value, "recall": .3, "background_predicted_spans": background},
                    "folds": [{"f1": f} for f in folds]}
        baseline = report("relevant_only", .2, 100, [.2] * 3)
        rows = [baseline, report("one_fold", .23, 50, [.5, .19, .19]), report("no_reduction", .24, 100, [.24] * 3)]
        self.assertEqual(choose(rows, policy), "relevant_only")
        rows.append(report("accepted", .21, 70, [.21] * 3))
        self.assertEqual(choose(rows, policy), "accepted")

    def test_config_rejects_nonfinite_or_invalid_selection_budgets(self):
        config = yaml.safe_load((ROOT / "configs/action_background_experiment.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            for value in (float("nan"), float("inf"), True, -.1, 1.1):
                config["selection"]["maximum_exact_recall_loss"] = value
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.subTest(value=value), self.assertRaises(ValueError):
                    read_config(path)

    def test_rejected_experiment_never_loads_dev_or_overwrites_active_model(self):
        self.run_synthetic(False)

    def test_winner_is_frozen_before_dev_and_saved_predictions_are_rechecked(self):
        self.run_synthetic(True)

    def run_synthetic(self, improving):
        config = yaml.safe_load((ROOT / "configs/action_background_experiment.yaml").read_text(encoding="utf-8"))
        data = SimpleNamespace(tokens=[[str(i)] for i in range(6)], tags=[["O"] for _ in range(6)],
                               relevant=[True, False] * 3, doc_ids=["a", "a", "b", "b", "c", "c"],
                               uids=[f"u{i}" for i in range(6)])
        if improving:
            data.tags = [["B-ACTION"] if i % 2 == 0 else ["O"] for i in range(6)]
        dev = SimpleNamespace(tokens=[["0"], ["1"]], tags=data.tags[:2], relevant=[True, False],
                              doc_ids=["d", "d"], uids=["d0", "d1"])
        folds = [([i for i in range(6) if i not in (n, n+1)], [n, n+1]) for n in (0, 2, 4)]
        prior = {"environment": {}, "source_sha256_lf": {}, "processed_train_sha256_lf": "digest",
                 "raw_dataset_sha256_lf": "digest", "config_sha256_lf": "digest"}
        records = [{"train_report_ids": sorted({data.doc_ids[i] for i in fit}),
                    "validation_report_ids": sorted({data.doc_ids[i] for i in valid}), "validation_sentences": len(valid)}
                   for fit, valid in folds]
        reference = {"provenance": prior, "folds": records, "candidates": [{"id": "action_boundaries", "parameters": config["parameters"],
                     "pooled": {"exact_strict": {"precision": .5 if improving else 0,
                     "recall": 1 if improving else 0, "f1": 2 / 3 if improving else 0}}}]}
        class FakeTagger:
            def fit(self, tokens, tags):
                return self
            def predict(self, tokens):
                return [["B-ACTION"] if improving and int(t[0]) % 2 == 0 else ["O"] for t in tokens]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("models/action_features_v1", "results/action_features_v1", "data/splits"):
                (root / name).mkdir(parents=True)
            active = root / "models/ner_action.joblib"
            active.write_bytes(b"synthetic active model")
            (root / "results/action_features_v1/cv.json").write_text(json.dumps(reference), encoding="utf-8")
            for name, ids in (("train", ["a", "b", "c"]), ("dev", ["d"]), ("test", ["t"])):
                (root / f"data/splits/{name}_report_ids.json").write_text(json.dumps(ids), encoding="utf-8")
            for n, record in enumerate(records, 1):
                predictions = [["B-ACTION" if improving else "O"] for _ in range(2)]
                cache = {"signature": {"fingerprint": _digest(prior), "candidate": {"id": "action_boundaries", "parameters": config["parameters"]},
                                       "fold": record}, "predictions": predictions, "predictions_sha256": _digest(predictions)}
                (root / f"models/action_features_v1/action_boundaries_fold_{n}.json").write_text(json.dumps(cache), encoding="utf-8")
            artifact = SimpleNamespace(manifest={"parameters": config["parameters"], "metadata": {"training_policy": "relevant_only",
                                      "selection": {"provenance": prior, "train_report_ids": ["a", "b", "c"]}}})
            artifact.tagger = SimpleNamespace(predict=lambda tokens: [["B-ACTION"] for _ in tokens])
            def train_only(**kwargs):
                split = kwargs["splits"]
                if split == ["dev"] and improving:
                    frozen = json.loads((root / "results/action_background_v1/selection.json").read_text(encoding="utf-8"))
                    self.assertEqual(frozen["selected"], "capped_background")
                    return {"dev": object()}
                self.assertEqual(split, ["train"])
                return {"train": object()}
            def load_artifact(path, **kwargs):
                return artifact if path == active else SimpleNamespace(tagger=FakeTagger())
            with patch("cti.experiment_action_background.ROOT", root), \
                    patch("cti.experiment_action_background.read_config", return_value=config), \
                    patch("cti.experiment_action_background.load_ner_tagger", side_effect=load_artifact), \
                    patch("cti.experiment_action_background.load_smah", side_effect=train_only) as loader, \
                    patch("cti.experiment_action_background.load_aligned_ner", side_effect=lambda layer, split, raw, processed: data if split == "train" else dev), \
                    patch("cti.experiment_action_background.get_report_grouped_splits", return_value=folds), \
                    patch("cti.experiment_action_background._environment", return_value={}), \
                    patch("cti.experiment_action_background.sha256_file", return_value="digest"), \
                    patch("cti.experiment_action_background.create_crf_tagger", side_effect=lambda **kw: FakeTagger()), \
                    patch("cti.experiment_action_background.save_ner_tagger", return_value={"artifact_sha256": "synthetic"}) as save:
                result = run(root / "synthetic.yaml")
            self.assertEqual(loader.call_count, 2 if improving else 1)
            self.assertEqual(result["selected"], "capped_background" if improving else "relevant_only")
            self.assertEqual(result["dev_evaluated"], improving)
            self.assertEqual(save.call_count, 1 if improving else 0)
            if improving:
                self.assertTrue(result["quantitative_checks_passed"])
                self.assertTrue(result["dev"]["reload_predictions_identical"])
            self.assertEqual(active.read_bytes(), b"synthetic active model")


if __name__ == "__main__":
    unittest.main()
