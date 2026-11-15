import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from cti.core.base import Document
from cti.core.pipeline import Pipeline
from cti.models.relevance import RelevanceFilter, SentenceRelevanceClassifier


ROOT = Path(__file__).resolve().parents[1]
TEXTS = [
    "APT29 deploys malware.", "APT29 deploys malware via a payload.",
    "Mimikatz steals passwords.", "Malware steals passwords via a payload.",
    "Read the privacy policy at help.example.org.",
    "Read the privacy policy on the website.",
    "Subscribe to the website newsletter.", "Contact website support for the newsletter.",
]
LABELS = [1, 1, 1, 1, 0, 0, 0, 0]


def fitted_model(**kwargs):
    return SentenceRelevanceClassifier(min_df=1, c=10, **kwargs).fit(TEXTS, LABELS)


class RelevanceClassifierTests(unittest.TestCase):
    def test_classifier_learns_relevance_and_handles_empty_predictions(self):
        model = fitted_model()
        self.assertEqual(model.predict([TEXTS[0], TEXTS[4]]), [True, False])
        self.assertEqual(model.predict([]), [])
        self.assertTrue(all(0 <= score <= 1 for score in model.predict_scores(TEXTS)))

    def test_fit_and_predict_share_normalization(self):
        model = fitted_model()
        scores = model.predict_scores([
            "  APT29\u200b  deploys malware at hxxps://evil[.]example[.]com. ",
            "APT29 deploys malware at https://evil.example.com.",
        ])
        self.assertEqual(scores[0], scores[1])

    def test_prediction_does_not_learn_development_vocabulary(self):
        model = fitted_model()
        before = dict(model.pipeline.named_steps["tfidf"].vocabulary_)
        model.predict_scores(["devonlymarker never seen during training"])
        self.assertEqual(model.pipeline.named_steps["tfidf"].vocabulary_, before)
        self.assertNotIn("devonlymarker", before)

    def test_threshold_override_does_not_mutate_saved_default(self):
        model = fitted_model(threshold=0.4)
        self.assertEqual(model.predict(TEXTS, threshold=0), [True] * len(TEXTS))
        self.assertEqual(model.predict(TEXTS, threshold=1), [False] * len(TEXTS))
        self.assertEqual(model.threshold, 0.4)

    def test_invalid_training_inputs_are_rejected(self):
        for texts, labels in (
            ([], []), (["malware"], []), (["malware"], [1]),
            (["malware", "privacy"], [1, "no"]),
            (["malware", None], [1, 0]), ("not a sentence list", [1, 0]),
        ):
            with self.subTest(texts=texts):
                with self.assertRaises((TypeError, ValueError)):
                    SentenceRelevanceClassifier(min_df=1).fit(texts, labels)

    def test_invalid_thresholds_are_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SentenceRelevanceClassifier(threshold=value)

    def test_unfitted_models_cannot_predict_or_save(self):
        model = SentenceRelevanceClassifier()
        with self.assertRaises(RuntimeError):
            model.predict(["malware"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                model.save(Path(directory) / "model.joblib")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_save_reload_preserves_scores_parameters_and_metadata(self):
        model = fitted_model(threshold=0.37)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            manifest = model.save(path, metadata={"training_split": "synthetic-train"})
            loaded = SentenceRelevanceClassifier.load(path)
        self.assertEqual(loaded.parameters, model.parameters)
        self.assertEqual(loaded.predict_scores(TEXTS), model.predict_scores(TEXTS))
        self.assertEqual(loaded.metadata, {"training_split": "synthetic-train"})
        self.assertEqual(loaded.artifact_sha256, manifest["artifact_sha256"])
        self.assertEqual(manifest["normalization"], "cleaner_defanger_v1")

    def test_corrupt_artifact_and_unknown_versions_are_rejected(self):
        model = fitted_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save(path)
            with path.open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash"):
                SentenceRelevanceClassifier.load(path)
            model.save(path)
            manifest_path = path.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact_version"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "version"):
                SentenceRelevanceClassifier.load(path)

    def test_invalid_metadata_does_not_write_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            with self.assertRaises(ValueError):
                fitted_model().save(path, metadata={"bad": float("nan")})
            self.assertFalse(path.exists())


class RelevancePipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "relevance.joblib"
        fitted_model().save(self.path)

    def config(self):
        config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
        config["pipeline"].insert(-1, {
            "module": "relevance_filter", "params": {"model_path": str(self.path)}
        })
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_pipeline_scores_sentences_and_retains_iocs_in_negative_sentences(self):
        pipeline = Pipeline.from_config(self.config())
        document = pipeline.run(TEXTS[0] + " " + TEXTS[4])
        sentences = document.metadata["sentences"]
        self.assertEqual([s["relevant"] for s in sentences], [True, False])
        self.assertEqual(document.metadata["relevance"]["relevant_sentence_count"], 1)
        self.assertEqual(len(document.metadata["relevance"]["artifact_sha256"]), 64)
        self.assertEqual(document.text, TEXTS[0] + " " + TEXTS[4])
        self.assertIn(("DOMAIN", "help.example.org"), {(e.label, e.text) for e in document.entities})
        for sentence in sentences:
            self.assertEqual(document.text[sentence["start"]:sentence["end"]], sentence["text"])

    def test_pipeline_does_not_require_spacy_or_neural_models(self):
        code = (
            "import sys\nfrom cti.core.pipeline import Pipeline\n"
            f"d = Pipeline.from_config({str(self.config())!r}).run('APT29 deploys malware.')\n"
            "assert d.metadata['relevance']['status'] == 'ok'\n"
            "assert not any(name in sys.modules for name in ('spacy', 'torch', 'transformers'))\n"
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_non_english_sentences_are_explicitly_unsupported(self):
        document = Document("हा हल्ला आहे", metadata={
            "language": "mr", "sentences": [{"text": "हा हल्ला आहे"}]
        })
        RelevanceFilter(self.path).process(document)
        self.assertEqual(document.metadata["relevance"]["status"], "unsupported_language")
        self.assertIsNone(document.metadata["sentences"][0]["relevant"])
        self.assertIsNone(document.metadata["sentences"][0]["relevance_score"])
        self.assertIsNone(document.metadata["relevance"]["relevant_sentence_count"])

    def test_empty_documents_and_missing_upstream_steps(self):
        module = RelevanceFilter(self.path)
        empty = module.process(Document("", metadata={"language": "en", "sentences": []}))
        self.assertEqual(empty.metadata["relevance"]["sentence_count"], 0)
        with self.assertRaisesRegex(ValueError, "sentence_splitter"):
            module.process(Document("malware"))

    def test_pipeline_threshold_can_override_artifact_threshold(self):
        document = Document(TEXTS[4], metadata={"language": "en", "sentences": [{"text": TEXTS[4]}]})
        RelevanceFilter(self.path, threshold=0).process(document)
        self.assertTrue(document.metadata["sentences"][0]["relevant"])


class RelevanceTrainingTests(unittest.TestCase):
    def test_training_uses_only_train_and_evaluates_only_dev(self):
        from cti.core.dataset import load_smah
        from cti.train_relevance import train_and_evaluate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, splits = root / "raw", root / "splits"
            raw.mkdir()
            splits.mkdir()
            records = []
            examples = {
                "train": list(zip(TEXTS, LABELS)),
                "dev": [("APT29 deploys malware devonlymarker.", 1), (TEXTS[4], 0)],
                "test": [("testonlymarker", 1)],
            }
            for split, items in examples.items():
                doc_id = f"synthetic-{split}"
                (splits / f"{split}_report_ids.json").write_text(json.dumps([doc_id]), encoding="utf-8")
                for index, (text, label) in enumerate(items):
                    records.append({
                        "uid": f"{split}-{index}", "doc_id": doc_id,
                        "sentence_text": text, "relevant": "yes" if label else "no",
                    })
            (raw / "SMAH-CTI.json").write_text(json.dumps(records), encoding="utf-8")
            output, report_path = root / "model.joblib", root / "report.json"
            with patch.dict(os.environ, {"CTI_ALLOW_TEST": "1"}):
                with patch("cti.train_relevance.load_smah", wraps=load_smah) as loader:
                    report = train_and_evaluate(raw, splits, output, report_path)
                    self.assertEqual(loader.call_args.kwargs["splits"], ["train", "dev"])
            loaded = SentenceRelevanceClassifier.load(output)
            vocabulary = loaded.pipeline.named_steps["tfidf"].vocabulary_
            self.assertNotIn("devonlymarker", vocabulary)
            self.assertNotIn("testonlymarker", vocabulary)
            self.assertEqual(report["metadata"]["training_sentences"], len(TEXTS))
            self.assertEqual(report["metadata"]["dev_sentences"], 2)
            self.assertFalse(report["metadata"]["test_split_used"])
            self.assertTrue(report["reload_predictions_identical"])
            with report_path.with_suffix(".predictions.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["uid"] for row in rows], ["dev-0", "dev-1"])
            # Leakage must fail before replacing the existing saved model.
            original = output.read_bytes()
            (splits / "dev_report_ids.json").write_text(json.dumps(["synthetic-train"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "share reports"):
                train_and_evaluate(raw, splits, output, report_path)
            self.assertEqual(output.read_bytes(), original)
            (splits / "dev_report_ids.json").write_text(json.dumps(["synthetic-dev"]), encoding="utf-8")
            (splits / "test_report_ids.json").write_text(json.dumps(["synthetic-train"]), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "Protected test reports"):
                train_and_evaluate(raw, splits, output, report_path)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
