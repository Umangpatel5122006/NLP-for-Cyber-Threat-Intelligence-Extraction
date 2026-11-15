import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cti.core.base import Document
from cti.core.sms_data import load_sms, read_rows, template_key, validate_split
from cti.extractors.message_classifier import MessageSpamClassifier
from cti.models.message_classifier import MessageClassifier, new_pipeline
from cti.train_message import train


HAM = ["Family lunch tomorrow bring food", "Meet family at lunch tomorrow", "Please bring lunch for family", "Tomorrow we meet for food"]
SPAM = ["Claim your free cash prize call now", "Win free cash reward call now", "Congratulations claim cash prize now", "Free lottery cash win claim prize"]


class MessageModelTests(unittest.TestCase):
    def fitted(self, profile="word_char_svm"):
        return new_pipeline(profile).fit(HAM + SPAM, ["ham"] * 4 + ["spam"] * 4)

    def test_classical_fit_predict_explanation_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sms.joblib"
            saved = MessageClassifier.save(self.fitted(), path, "word_char_svm", {"dev_metrics": {}})
            restored = MessageClassifier.load(path)
            for text, expected in [(HAM[0], "ham"), (SPAM[0], "spam")]:
                result = restored.predict_one(text)
                self.assertEqual(result["label"], expected)
                self.assertEqual(result, saved.predict_one(text))
                self.assertEqual(result["score_kind"], "decision_margin")
                self.assertTrue(result["supporting_features"])
                self.assertNotIn("probability", result)

    def test_tampered_hash_or_runtime_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sms.joblib"
            MessageClassifier.save(self.fitted(), path, "word_char_svm", {})
            sidecar = path.with_suffix(".manifest.json")
            original = json.loads(sidecar.read_text())
            for key in ("hash", "environment"):
                changed = json.loads(json.dumps(original))
                if key == "hash":
                    changed["artifact_sha256"] = "wrong"
                else:
                    changed["environment"]["python"] = "0.0"
                sidecar.write_text(json.dumps(changed))
                with patch("cti.models.message_classifier.joblib.load") as deserialize:
                    with self.assertRaises(ValueError):
                        MessageClassifier.load(path)
                    deserialize.assert_not_called()

    def test_training_vocabulary_does_not_learn_prediction_text(self):
        with tempfile.TemporaryDirectory() as directory:
            model = MessageClassifier.save(self.fitted(), Path(directory) / "sms.joblib", "word_char_svm", {})
            vocab = dict(model.pipeline["features"].transformer_list[0][1].vocabulary_)
            model.predict_one("unseenxyz uniqueworddef nevertrainedabc")
            self.assertEqual(vocab, model.pipeline["features"].transformer_list[0][1].vocabulary_)
            self.assertNotIn("unseenxyz", vocab)

    def test_missing_model_degrades_without_inventing_prediction_and_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sms.joblib"
            module = MessageSpamClassifier(path)
            doc = Document(SPAM[0]); doc.metadata["language"] = "en"
            self.assertEqual(module.process(doc).metadata["message_classifier"]["status"], "unavailable")
            MessageClassifier.save(self.fitted(), path, "word_char_svm", {})
            self.assertTrue(module.readiness()["ready"])
            self.assertEqual(module.process(doc).metadata["message_classifier"]["label"], "spam")

    def test_out_of_scope_messages_do_not_load_or_run_sms_model(self):
        module = MessageSpamClassifier("missing.joblib", max_characters=50)
        for text, language in [("x" * 51, "en"), ("message", "hi"), (" ", "en")]:
            doc = Document(text); doc.metadata["language"] = language
            with patch.object(module, "readiness", side_effect=AssertionError("must not load")):
                self.assertEqual(module.process(doc).metadata["message_classifier"]["status"], "not_applicable")

    def test_template_groups_remove_numeric_link_and_punctuation_variants(self):
        self.assertEqual(template_key("WIN £500! Visit https://one.example/x"), template_key("win £900. visit https://two.example/y"))
        self.assertNotEqual(template_key("Family lunch tomorrow"), template_key("Cash prize winner"))

    def test_optional_model_failure_is_reported_without_breaking_other_detectors(self):
        module = MessageSpamClassifier("missing.joblib")
        doc = Document("A message for review."); doc.metadata["language"] = "en"
        with patch.object(module, "readiness", return_value={"ready": True}):
            module.artifact = type("Broken", (), {"predict_one": lambda *args: (_ for _ in ()).throw(RuntimeError("private error"))})()
            result = module.process(doc).metadata["message_classifier"]
        self.assertEqual(result["code"], "prediction_failed")
        self.assertNotIn("private error", json.dumps(result))

    def test_split_validation_rejects_leakage_and_duplicate_ids(self):
        rows = [{"id": "a", "group": "same"}, {"id": "b", "group": "same"}, {"id": "c", "group": "other"}]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split(rows, {"partitions": {"train": ["a"], "dev": ["b"], "test": ["c"]}})
        with self.assertRaises(ValueError):
            validate_split(rows, {"partitions": {"train": ["a"], "dev": ["a", "b"], "test": ["c"]}})

    def test_sms_test_lock_precedes_file_access(self):
        with patch("cti.core.sms_data.Path.read_text", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError, "reserved"):
                load_sms("test", "nonexistent", "nonexistent")

    def test_training_cli_function_selects_on_train_and_persists_dev_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = []
            for index in range(36):
                label = "ham" if index % 2 == 0 else "spam"
                text = (HAM[index % 4] if label == "ham" else SPAM[index % 4]) + " token" + chr(97 + index // 2)
                rows.append({"id": f"sms-{index:05d}", "text": text, "label": label, "group": template_key(text)})
            partitions = {"train": [r["id"] for r in rows[:24]], "dev": [r["id"] for r in rows[24:30]], "test": [r["id"] for r in rows[30:]]}
            seen = []
            def synthetic_load(split, *args):
                seen.append(split)
                if split == "test":
                    raise AssertionError("Test access forbidden")
                return [r for r in rows if r["id"] in partitions[split]]
            data, split_path = path / "data.txt", path / "split.json"
            data.write_text("synthetic provenance")
            split_path.write_text(json.dumps(partitions))
            with patch("cti.train_message.load_sms", side_effect=synthetic_load), contextlib.redirect_stdout(io.StringIO()):
                report = train(data, split_path, path / "model.joblib", path / "results")
            self.assertEqual(seen, ["train", "dev"])
            self.assertFalse(report["selection"]["dev_used_for_selection"])
            self.assertFalse(report["test_evaluated"])
            self.assertTrue(report["reload_predictions_identical"])


if __name__ == "__main__":
    unittest.main()
