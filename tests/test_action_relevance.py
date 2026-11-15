"""Synthetic gate/selection/cache tests; no real protected annotations loaded."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import yaml

from cti.experiment_action import _digest
from cti.experiment_action_relevance import cache_predictions, checks, choose, gate, read_config, run, score


ROOT = Path(__file__).resolve().parents[1]


class ActionRelevanceTests(unittest.TestCase):
    def test_gate_removes_background_without_changing_kept_tokens_or_input(self):
        tags = [["B-ACTION", "I-ACTION"], ["B-ACTION", "O"]]
        original = copy.deepcopy(tags)
        self.assertEqual(gate(tags, [0.2, 0.19], 0.2), [tags[0], ["O", "O"]])
        self.assertEqual(gate(tags, [0.2, 0.19], None), tags)
        self.assertEqual(tags, original)

    def test_gate_rejects_misaligned_or_nonfinite_scores(self):
        for values in ([], [float("nan")], [float("inf")], [-0.1], [1.1], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                gate([["O"]], values, 0.2)

    def test_exact_span_counts_and_relevant_coverage_have_distinct_denominators(self):
        gold = [["B-ACTION", "I-ACTION"], ["O", "O"]]
        predicted = [["B-ACTION", "I-ACTION"], ["B-ACTION", "O"]]
        baseline = score(gold, predicted, [True, False], [0.9, 0.1], None)
        filtered = score(gold, predicted, [True, False], [0.9, 0.1], 0.2)
        self.assertEqual((baseline["tp"], baseline["gold_spans"], baseline["predicted_spans"]), (1, 1, 2))
        self.assertAlmostEqual(baseline["f1"], 2 / 3)
        self.assertEqual(filtered["f1"], 1.0)
        self.assertEqual(filtered["background_predicted_spans"], 0)
        self.assertEqual(filtered["relevant_sentence_skip_rate"], 0)

    def test_strict_iob2_scoring_does_not_repair_an_illegal_inside_start(self):
        result = score([["B-ACTION"]], [["I-ACTION"]], [True], [0.9], None)
        self.assertEqual((result["tp"], result["predicted_spans"], result["gold_spans"]), (0, 0, 1))

    def test_selection_rejects_precision_gain_that_exceeds_recall_or_coverage_budget(self):
        policy = {"minimum_f1_gain": 0.005, "minimum_fold_wins": 2,
                  "maximum_exact_recall_loss": 0.01, "maximum_relevant_sentence_skip_rate": 0.03}
        def report(name, f1, recall, skipped, cutoff):
            return {"id": name, "cutoff": cutoff, "pooled": {"f1": f1, "recall": recall, "relevant_sentence_skip_rate": skipped},
                    "folds": [{"f1": f1}] * 3}
        rows = [report("no_gate", 0.2, 0.3, 0, None), report("recall_loss", 0.4, 0.2, 0.01, 0.5),
                report("coverage_loss", 0.3, 0.3, 0.2, 0.3), report("conservative", 0.21, 0.3, 0.01, 0.2)]
        self.assertEqual(choose(rows, policy)["id"], "conservative")
        self.assertFalse(rows[1]["eligible"])
        self.assertFalse(rows[2]["eligible"])
        self.assertEqual(choose(rows[:3], policy)["id"], "no_gate")

    def test_dev_checks_do_not_override_predeclared_recall_budget(self):
        baseline = {"f1": 0.2, "recall": 0.3}
        candidate = {"f1": 0.4, "recall": 0.25, "relevant_sentence_skip_rate": 0.01}
        policy = {"minimum_f1_gain": 0.002, "maximum_exact_recall_loss": 0.01, "maximum_relevant_sentence_skip_rate": 0.03}
        self.assertFalse(checks(baseline, candidate, policy)["exact_recall"])

    def test_cutoff_configuration_rejects_nan_duplicates_and_boolean_values(self):
        config = yaml.safe_load((ROOT / "configs/action_relevance_experiment.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "config.yaml"
            for values in ([float("nan")], [0.2, 0.2], [True], [1.0], []):
                config["cutoffs"] = values
                p.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.subTest(values=values), self.assertRaises(ValueError):
                    read_config(p)

    def test_action_cache_checks_provenance_digest_and_token_alignment(self):
        signature = {"fold": "synthetic"}
        cache = {"signature": signature, "predictions": [["B-ACTION"]], "predictions_sha256": _digest([["B-ACTION"]])}
        self.assertEqual(cache_predictions(cache, signature, [["run"]]), [["B-ACTION"]])
        for changed, expected_signature, tokens in (({**cache, "predictions_sha256": "bad"}, signature, [["run"]]),
                                                    (cache, {"fold": "other"}, [["run"]]),
                                                    (cache, signature, [["run", "now"]])):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                cache_predictions(changed, expected_signature, tokens)

    def test_baseline_selection_never_loads_dev_or_predicts_with_full_action_model(self):
        config = yaml.safe_load((ROOT / "configs/action_relevance_experiment.yaml").read_text(encoding="utf-8"))
        rows = pd.DataFrame([{"uid": f"u{i}", "sentence_text": f"Synthetic sentence {i}"} for i in range(6)])
        data = SimpleNamespace(tokens=[["word"] for _ in range(6)], tags=[["O"] for _ in range(6)],
                               uids=[f"u{i}" for i in range(6)], doc_ids=[f"r{i}" for i in range(6)],
                               relevant=[bool(i % 2) for i in range(6)])
        folds = [([i for i in range(6) if i not in (n, n+1)], [n, n+1]) for n in (0, 2, 4)]
        prior = {"environment": {}, "source_sha256_lf": {}, "processed_train_sha256_lf": "digest",
                 "raw_dataset_sha256_lf": "digest", "config_sha256_lf": "digest"}
        params = {"c1": 1.0}
        fold_records = [{"train_report_ids": sorted(data.doc_ids[i] for i in fit),
                         "validation_report_ids": sorted(data.doc_ids[i] for i in valid), "validation_sentences": len(valid)}
                        for fit, valid in folds]
        reference = {"provenance": prior, "folds": fold_records, "candidates": [{"id": "action_boundaries", "parameters": params,
                     "pooled": {"exact_strict": {"precision": 0, "recall": 0, "f1": 0}}}]}
        class FakeRelevance:
            def __init__(self, **kwargs):
                pass
            def fit(self, texts, labels):
                assert set(labels) == {0, 1}
                return self
            def predict_scores(self, texts):
                return [0.9] * len(texts)
        def train_only(**kwargs):
            self.assertEqual(kwargs["splits"], ["train"])
            return {"train": rows}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache_folder = root / "models/action_features_v1"
            cache_folder.mkdir(parents=True)
            output = root / "results/action_features_v1"
            output.mkdir(parents=True)
            (output / "cv.json").write_text(json.dumps(reference), encoding="utf-8")
            active = root / "models/ner_action.joblib"
            active.write_bytes(b"synthetic model marker")
            for n, fold in enumerate(fold_records, 1):
                predictions = [["O"], ["O"]]
                cache = {"signature": {"fingerprint": _digest(prior), "candidate": {"id": "action_boundaries", "parameters": params}, "fold": fold},
                         "predictions": predictions, "predictions_sha256": _digest(predictions)}
                (cache_folder / f"action_boundaries_fold_{n}.json").write_text(json.dumps(cache), encoding="utf-8")
            artifact = SimpleNamespace(manifest={"parameters": params, "metadata": {"training_policy": "relevant_only",
                                       "selection": {"train_report_ids": data.doc_ids, "provenance": prior}}})
            with patch("cti.experiment_action_relevance.ROOT", root), \
                    patch("cti.experiment_action_relevance.read_config", return_value=config), \
                    patch("cti.experiment_action_relevance.load_ner_tagger", return_value=artifact), \
                    patch("cti.experiment_action_relevance.load_smah", side_effect=train_only) as loader, \
                    patch("cti.experiment_action_relevance.load_aligned_ner", return_value=data), \
                    patch("cti.experiment_action_relevance.get_report_grouped_splits", return_value=folds), \
                    patch("cti.experiment_action_relevance._environment", return_value={}), \
                    patch("cti.experiment_action_relevance.sha256_file", return_value="digest"), \
                    patch("cti.experiment_action_relevance.SentenceRelevanceClassifier", FakeRelevance):
                result = run(root / "synthetic.yaml")
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(result["selected"], "no_gate")
            self.assertFalse(result["dev_evaluated"])
            self.assertFalse(result["test_evaluated"])
            self.assertEqual(hashlib.sha256(active.read_bytes()).hexdigest(), hashlib.sha256(b"synthetic model marker").hexdigest())


if __name__ == "__main__":
    unittest.main()
