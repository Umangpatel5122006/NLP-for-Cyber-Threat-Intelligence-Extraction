"""Synthetic ending-feature, leakage, persistence and selection tests."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from cti.experiment_action_background import metrics
from cti.experiment_action_ending import checks, choose, fit, read_config, run
from cti.models.action_end_crf import (ActionEndCRFTagger, edge_counts, ending_features,
                                      load_candidate, save_candidate)
from cti.models.ner_artifact import load_ner_tagger


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {"c1": 0, "c2": .01, "max_iterations": 15,
              "feature_config": {"use_action_morphology": True, "use_action_lexicon": True}}


class EndingFeaturesTests(unittest.TestCase):
    def test_structural_cues_preserve_base_and_do_not_assign_labels(self):
        words = ["exfiltration", ",", "while", "running"]
        base = [{"original": i} for i in range(4)]
        original = copy.deepcopy(base)
        result = ending_features(words, base)
        self.assertTrue(result[0]["ending:nominal_suffix"])
        self.assertTrue(result[0]["ending:next_punct"])
        self.assertTrue(result[1]["ending:next_clause"])
        self.assertEqual(base, original)
        self.assertTrue(all(row["original"] == i for i, row in enumerate(result)))

    def test_current_sentence_cannot_create_its_own_known_edge_statistics(self):
        words = ["repeat"] * 4
        matrices = []
        for labels in (["B-ACTION"] * 4, ["O"] * 4):
            counts = edge_counts([words], [labels])
            matrices.append(ending_features(words, [{} for _ in words], counts, counts))
        self.assertEqual(matrices[0], matrices[1])
        self.assertFalse(matrices[0][0]["ending:edge:right:known"])

    def test_held_out_prediction_does_not_update_learned_edge_dictionary(self):
        model = ActionEndCRFTagger(learned_edges=True, **PARAMETERS)
        tokens = [["steal", "data"], ["plain", "text"]] * 3
        tags = [["B-ACTION", "I-ACTION"], ["O", "O"]] * 3
        model.fit(tokens, tags)
        before = copy.deepcopy(model.edge_lexicon_)
        result = model.predict([[], ["novel", "destination"], []])
        self.assertEqual([len(row) for row in result], [0, 2, 0])
        self.assertEqual(model.edge_lexicon_, before)

    def test_exclusion_uses_other_sentence_counts_and_rejects_impossible_counts(self):
        words, labels = ["steal", "data"], ["B-ACTION", "O"]
        full = edge_counts([words] * 4, [labels] * 4)
        excluded = edge_counts([words], [labels])
        result = ending_features(words, [{}, {}], full, excluded)
        self.assertTrue(result[0]["ending:edge:right:known"])
        self.assertAlmostEqual(result[1]["ending:edge:left:end_rate"], 3.5 / 4)
        with self.assertRaises(ValueError):
            ending_features(words, [{}, {}], {}, excluded)

    def test_candidate_roundtrip_preserves_both_profiles_and_gold_inputs(self):
        tokens = [["initial", "access"], ["plain", "text"]] * 3
        tags = [["B-ACTION", "I-ACTION"], ["O", "O"]] * 3
        original = copy.deepcopy(tags)
        with tempfile.TemporaryDirectory() as folder:
            for edges in (False, True):
                model = ActionEndCRFTagger(learned_edges=edges, **PARAMETERS).fit(tokens, tags)
                path = Path(folder) / f"candidate_{edges}.joblib"
                manifest = save_candidate(model, path, {"synthetic": True})
                self.assertEqual(manifest["metadata"]["ending_profile"]["learned_edges"], edges)
                self.assertEqual(model.predict(tokens), load_candidate(path).tagger.predict(tokens))
                with patch("cti.models.action_end_crf.sha256_file", return_value="changed"):
                    with self.assertRaisesRegex(ValueError, "source changed"):
                        load_ner_tagger(path, expected_layer="action")
        self.assertEqual(tags, original)

    def test_invalid_training_and_unfitted_predictions_fail_explicitly(self):
        model = ActionEndCRFTagger(**PARAMETERS)
        with self.assertRaises(RuntimeError):
            model.predict([["word"]])
        for tokens, tags in (([[]], [[]]), ([["word"]], [["B-MALWARE_TOOL"]]), ([["word"]], [[]])):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                model.fit(tokens, tags)
        with self.assertRaises(ValueError):
            model.fit([["word"]], [["O"]], dev_tokens=[["secret"]], dev_tags=[["O"]])
        with self.assertRaises(ValueError):
            ActionEndCRFTagger(learned_edges=1, **PARAMETERS)


class EndingExperimentTests(unittest.TestCase):
    def test_selection_requires_gain_recall_background_budget_and_multiple_wins(self):
        policy = {"minimum_f1_gain": .005, "maximum_exact_recall_loss": .01,
                  "maximum_background_increase": .05, "minimum_fold_wins": 2}
        def report(name, f1, recall=.3, bg=100, folds=None):
            return {"id": name, "pooled": {"f1": f1, "recall": recall, "background_predicted_spans": bg},
                    "folds": [{"f1": x} for x in (folds or [f1] * 3)]}
        rows = [report("baseline", .2), report("lost_recall", .3, .28), report("background", .3, bg=106),
                report("one_win", .3, folds=[.5, .1, .1]), report("small_gain", .201)]
        self.assertEqual(choose(rows, policy), "baseline")
        rows.append(report("accepted", .21))
        self.assertEqual(choose(rows, policy), "accepted")
        self.assertFalse(checks(rows[0]["pooled"], rows[1]["pooled"], policy)["exact_recall"])

    def test_config_rejects_nonfinite_budgets_and_integer_feature_flags(self):
        original = yaml.safe_load((ROOT / "configs/action_ending_experiment.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            for value in (float("nan"), float("inf"), True, -.1):
                config = copy.deepcopy(original)
                config["selection"]["maximum_background_increase"] = value
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_config(path)
            config = copy.deepcopy(original)
            config["candidates"][0]["learned_edges"] = 0
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_config(path)

    def test_fit_helper_excludes_held_out_and_background_sentences(self):
        data = SimpleNamespace(tokens=[["fit"], ["background"], ["heldout"]], tags=[["O"]] * 3, relevant=[True, False, True])
        model = SimpleNamespace(fit=lambda tokens, tags: self.assertEqual(tokens, [["fit"]]))
        with patch("cti.experiment_action_ending.ActionEndCRFTagger", return_value=model):
            self.assertIs(fit({"learned_edges": False}, data, [0, 1], PARAMETERS), model)

    def test_rejected_run_does_not_load_dev_or_replace_active_artifact(self):
        self.run_synthetic(False)

    def test_qualifying_winner_is_frozen_before_dev_and_reload_check(self):
        self.run_synthetic(True)

    def run_synthetic(self, improving):
        config = yaml.safe_load((ROOT / "configs/action_ending_experiment.yaml").read_text(encoding="utf-8"))
        train = SimpleNamespace(tokens=[[str(i)] for i in range(6)], tags=[["B-ACTION"] if improving and i % 2 == 0 else ["O"] for i in range(6)],
            relevant=[True, False] * 3, doc_ids=["a", "a", "b", "b", "c", "c"], uids=[f"u{i}" for i in range(6)])
        dev = SimpleNamespace(tokens=[["0"], ["1"]], tags=train.tags[:2], relevant=[True, False], doc_ids=["d", "d"], uids=["d0", "d1"])
        baseline = [["B-ACTION" if improving else "O"] for _ in range(6)]
        folds = [([i for i in range(6) if i not in (n, n+1)], [n, n+1]) for n in (0, 2, 4)]
        records = [{"train_report_ids": sorted({train.doc_ids[i] for i in a}), "validation_report_ids": sorted({train.doc_ids[i] for i in b}),
                    "validation_sentences": len(b)} for a, b in folds]
        class FakeModel:
            def predict(self, tokens):
                return [["B-ACTION"] if improving and int(t[0]) % 2 == 0 else ["O"] for t in tokens]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("data/splits", "models/action_background_v1", "results/action_background_v1", "configs"):
                (root / name).mkdir(parents=True)
            active_path = root / "models/ner_action.joblib"
            active_path.write_bytes(b"synthetic active")
            prior = {"environment": {}, "source_sha256_lf": {}, "reference_cache_sha256_lf": {},
                     "unchanged_files_sha256": {"models/ner_action.joblib": hashlib_sha(active_path)},
                     "split_sha256_lf": {s: "digest" for s in ("train", "dev", "test")},
                     "raw_sha256_lf": "digest", "processed_train_sha256_lf": "digest", "config_sha256_lf": "digest"}
            reference = {"id": "relevant_only", "pooled": metrics(train, baseline), "folds": [metrics(train, baseline, b) for _, b in folds]}
            values = {"summary.json": {"selected": "relevant_only", "provenance": prior},
                      "selection.json": {"provenance": prior, "train_report_ids": ["a", "b", "c"], "candidates": [reference]},
                      "integrity_audit.json": {"passed": True, "audit_script_sha256_lf": "digest", "oof_sha256_lf": {"models/action_background_v1/relevant_only_oof.json": "digest"}},
                      "cv.json": {"folds": records}}
            for name, value in values.items():
                (root / "results/action_background_v1" / name).write_text(json.dumps(value), encoding="utf-8")
            (root / "models/action_background_v1/relevant_only_oof.json").write_text(json.dumps(baseline), encoding="utf-8")
            (root / "configs/action_background_experiment.yaml").write_text(yaml.safe_dump({"parameters": config["parameters"]}), encoding="utf-8")
            for split, ids in (("train", ["a", "b", "c"]), ("dev", ["d"]), ("test", ["t"])):
                (root / f"data/splits/{split}_report_ids.json").write_text(json.dumps(ids), encoding="utf-8")
            artifact = SimpleNamespace(manifest={"parameters": config["parameters"], "metadata": {"training_policy": "relevant_only"}},
                                       tagger=SimpleNamespace(predict=lambda tokens: [["B-ACTION"] for _ in tokens]))
            def load_split(**kwargs):
                split = kwargs["splits"]
                if split == ["dev"] and improving:
                    decision = json.loads((root / "results/action_ending_v1/selection.json").read_text(encoding="utf-8"))
                    self.assertEqual(decision["selected"], "ending_cues")
                    return {"dev": object()}
                self.assertEqual(split, ["train"])
                return {"train": object()}
            with patch("cti.experiment_action_ending.ROOT", root), \
                    patch("cti.experiment_action_ending.read_config", return_value=config), \
                    patch("cti.experiment_action_ending._environment", return_value={}), \
                    patch("cti.experiment_action_ending.sha256_file", return_value="digest"), \
                    patch("cti.experiment_action_ending.load_ner_tagger", return_value=artifact), \
                    patch("cti.experiment_action_ending.load_smah", side_effect=load_split) as loader, \
                    patch("cti.experiment_action_ending.load_aligned_ner", side_effect=lambda layer, split, raw, path: train if split == "train" else dev), \
                    patch("cti.experiment_action_ending.get_report_grouped_splits", return_value=folds), \
                    patch("cti.experiment_action_ending.fit", return_value=FakeModel()), \
                    patch("cti.experiment_action_ending.save_candidate", return_value={"artifact_sha256": "synthetic"}), \
                    patch("cti.experiment_action_ending.load_candidate", return_value=SimpleNamespace(tagger=FakeModel())):
                result = run(root / "synthetic.yaml")
            self.assertEqual(loader.call_count, 2 if improving else 1)
            self.assertEqual(result["selected"], "ending_cues" if improving else "baseline")
            self.assertEqual(result["dev_evaluated"], improving)
            self.assertEqual(active_path.read_bytes(), b"synthetic active")
            if improving:
                self.assertTrue(result["quantitative_checks_passed"])


def hashlib_sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
