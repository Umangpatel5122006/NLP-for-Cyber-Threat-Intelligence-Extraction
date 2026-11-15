import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cti.core.sms_domain import audit_overlap, canonical, group_rows, load_domain
from cti.experiment_message import activate, choose, promotion_checks
from cti.models.message_classifier import MessageClassifier, new_pipeline
from cti.models.message_features import MessageConcepts, normalize_message
from cti.models.message_v2 import ExpandedMessageClassifier, load_message_classifier, new_expanded_pipeline
from cti.web.server import ReviewEngine


TEXTS = ["Family dinner tomorrow bring food", "Please bring food for family dinner",
         "Meet the family for lunch tomorrow", "Your transaction has been credited no reply needed",
         "Claim your free cash prize reply now", "Government grant giving money call to claim",
         "Share your banking password to receive cash", "Win cash prize send registration fee"]
LABELS = ["ham"] * 4 + ["spam"] * 4


class MessageExpansionTests(unittest.TestCase):
    def test_fixed_normalization_handles_number_and_url_variations(self):
        a = "Claim INR 1500000. SMS 6969696969 at https://one.example/payout"
        b = "Claim INR 42. SMS 1234 at https://two.example/offer"
        self.assertEqual(normalize_message(a), normalize_message(b))
        self.assertEqual(normalize_message("\ufeffClick&#32;hxxps://evil[.]example/x"), "click urltoken")
        self.assertNotIn("\ufffd", normalize_message("OCR\ufffd text"))

    def test_concepts_are_label_independent_and_do_not_encode_numbers_or_verdicts(self):
        a = MessageConcepts.describe("Government giving 15 lakhs rupees. Reply with bank account.")
        b = MessageConcepts.describe("Government giving 42 lakhs rupees. Reply with bank account.")
        self.assertEqual(a, b)
        self.assertEqual(a["money_and_authority"], 1.0)
        self.assertNotIn("spam", a)
        self.assertNotIn("rule_status", a)

    def test_expanded_classifier_learns_label_weights_instead_of_a_fixed_verdict(self):
        model = new_expanded_pipeline("concept_svm").fit(TEXTS, LABELS)
        flipped = new_expanded_pipeline("concept_svm").fit(TEXTS, ["spam" if y == "ham" else "ham" for y in LABELS])
        self.assertEqual(model.predict([TEXTS[4]]).tolist(), ["spam"])
        self.assertEqual(flipped.predict([TEXTS[4]]).tolist(), ["ham"])

    def test_legacy_and_expanded_artifacts_round_trip_through_dispatcher(self):
        with tempfile.TemporaryDirectory() as directory:
            for profile in ("word_char_svm", "normalized_svm", "concept_svm"):
                path = Path(directory) / (profile + ".joblib")
                factory = new_pipeline if profile == "word_char_svm" else new_expanded_pipeline
                saver = MessageClassifier if profile == "word_char_svm" else ExpandedMessageClassifier
                model = factory(profile).fit(TEXTS, LABELS)
                saved = saver.save(model, path, profile, {"dev_metrics": {}})
                loaded = load_message_classifier(path)
                self.assertEqual(loaded.predict_one(TEXTS[4]), saved.predict_one(TEXTS[4]))
                self.assertEqual(loaded.manifest["artifact_version"], 1 if profile == "word_char_svm" else 2)

    def test_expanded_feature_drift_rejected_before_deserializing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.joblib"
            model = new_expanded_pipeline("concept_svm").fit(TEXTS, LABELS)
            ExpandedMessageClassifier.save(model, path, "concept_svm", {})
            sidecar = path.with_suffix(".manifest.json")
            record = json.loads(sidecar.read_text())
            record["environment"]["expanded_sources_sha256"]["message_features.py"] = "changed"
            sidecar.write_text(json.dumps(record))
            with patch("cti.models.message_v2.joblib.load") as deserialize:
                with self.assertRaises(ValueError):
                    load_message_classifier(path)
                deserialize.assert_not_called()

    def test_prediction_does_not_fit_new_concepts_or_vocabulary(self):
        model = new_expanded_pipeline("concept_svm").fit(TEXTS, LABELS)
        words = dict(model["features"].transformer_list[0][1].vocabulary_)
        names = model["features"].get_feature_names_out().tolist()
        model.predict(["newunseenxyz account some completely different words"])
        self.assertEqual(words, model["features"].transformer_list[0][1].vocabulary_)
        self.assertEqual(names, model["features"].get_feature_names_out().tolist())

    def test_exact_and_near_uci_matches_are_excluded_without_protected_labels(self):
        protected = ["Your banking account has been suspended. Please verify your details immediately.",
                     "Meet me for dinner at 7 tonight"]
        rows = [{"id": "exact", "text": "Meet me for dinner at 9 tonight", "original_label": "ham"},
                {"id": "near", "text": "Your banking acount has been suspended. Please verify your details immediately.", "original_label": "smishing"},
                {"id": "new", "text": "The library meeting has moved to a different room in the west building", "original_label": "ham"}]
        retained, audit = audit_overlap(rows, protected)
        self.assertEqual([r["id"] for r in retained], ["new"])
        self.assertEqual(audit["exact_excluded"], 1)
        self.assertEqual(audit["near_excluded"], 1)
        self.assertFalse(audit["protected_labels_used"])

    def test_near_duplicates_share_a_group(self):
        rows = group_rows([
            {"id": "a", "text": "Your banking account has been suspended. Please verify your details immediately."},
            {"id": "b", "text": "Your banking acount has been suspended. Please verify your details immediately."},
            {"id": "c", "text": "Bring the milk and bread for the family dinner tonight"}])
        self.assertEqual(rows[0]["group"], rows[1]["group"])
        self.assertNotEqual(rows[0]["group"], rows[2]["group"])

    def test_domain_test_lock_precedes_any_file_access(self):
        with patch("cti.core.sms_domain.Path.read_text", side_effect=AssertionError("no reads")):
            with self.assertRaisesRegex(ValueError, "reserved"):
                load_domain("test")

    def test_cv_selection_enforces_false_positive_budget(self):
        def row(name, mean, fp):
            return {"id": name, "pooled": {"source_mean_f1": mean, "per_source": {"uci": {"f1": 0.95, "fp": fp}}},
                    "folds": [{"source_mean_f1": mean}] * 3}
        rows = [row("baseline", 0.8, 2), row("too_many_false_alarms", 0.99, 20), row("good", 0.9, 4)]
        policy = {"minimum_source_mean_f1_gain": 0.005, "minimum_fold_wins": 2,
                  "maximum_uci_f1_loss": 0.01, "maximum_extra_uci_false_positives": 5}
        self.assertEqual(choose(rows, policy), "good")
        self.assertFalse(rows[1]["eligible"])

    def test_dev_gates_do_not_accept_sensitivity_gain_with_excessive_false_alarms(self):
        baseline = {"source_mean_f1": 0.8, "per_source": {"uci": {"f1": 0.98, "fp": 0}, "mendeley": {"f1": 0.64}}}
        candidate = {"source_mean_f1": 0.9, "per_source": {"uci": {"f1": 0.97, "fp": 8}, "mendeley": {"f1": 0.9}}}
        policy = {"maximum_uci_dev_f1_loss": 0.005, "maximum_extra_uci_dev_false_positives": 2,
                  "minimum_domain_dev_f1_gain": 0.02, "minimum_dev_source_mean_f1_gain": 0.005}
        checks = promotion_checks(baseline, candidate, policy)
        self.assertFalse(checks["uci_f1"])
        self.assertFalse(checks["uci_false_positives"])
        self.assertTrue(checks["domain_f1"])

    def test_release_hold_blocks_default_replacement_before_any_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "summary.json").write_text(json.dumps({
                "eligible_for_activation": True, "deployment_hold": "diagnostic_false_alarms"}))
            with patch("cti.experiment_message.shutil.copy2") as copy:
                with self.assertRaisesRegex(ValueError, "held"):
                    activate(directory)
                copy.assert_not_called()

    def test_api_candidate_preview_uses_its_own_artifact_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.joblib"
            model = new_expanded_pipeline("normalized_svm").fit(TEXTS, LABELS)
            artifact = ExpandedMessageClassifier.save(model, path, "normalized_svm", {
                "training_protocol": "synthetic_preview", "training_sources": ["synthetic"], "dev_metrics": {}})
            engine = ReviewEngine(ioc_only=True, message_model_path=path)
            with patch("pathlib.Path.write_bytes", side_effect=AssertionError("inference must not write")):
                result = engine.analyze({"mode": "ioc", "text": TEXTS[4]})
            details = result["document"]["metadata"]["message_classifier"]
            self.assertEqual(details["artifact_sha256"], artifact.manifest["artifact_sha256"])
            self.assertEqual(details["training_protocol"], "synthetic_preview")
            self.assertTrue(any(row["kind"] == "message_model" for row in result["document"]["review_findings"]))


if __name__ == "__main__":
    unittest.main()
