"""Synthetic task separation and coverage checks; no probability is inferred."""

import unittest
from types import SimpleNamespace

from cti.core.base import Document
from cti.web.tasks import summarize_tasks


def document():
    doc = Document("Synthetic source text")
    doc.metadata.update(language="en", ner={"entities": {"status": "ok"}, "action": {"status": "ok"}},
        message_assessment={"status": "no_pattern_detected", "summary": "No supported detection; safety is not established."},
        message_risk={"status": "no_pattern_detected"}, message_classifier={"status": "ok", "label": "ham"})
    return doc


class TaskSummaryTests(unittest.TestCase):
    def test_positive_message_is_visible_when_no_entities_or_iocs_exist(self):
        doc = document()
        doc.metadata["message_assessment"].update(status="suspicious", summary="Suspicious request.")
        doc.metadata["message_risk"]["status"] = "suspicious"
        doc.review_findings = [{"kind": "message_rule"}]
        tasks = summarize_tasks(doc, "ner", ioc_checked=True)
        self.assertEqual(tasks["message"]["status"], "suspicious")
        self.assertEqual(tasks["message"]["finding_count"], 1)
        self.assertEqual(tasks["extraction"]["primary_findings"], 0)
        self.assertEqual(tasks["message"]["model_label"], "ham")

    def test_extracted_indicators_do_not_become_a_message_threat_verdict(self):
        doc = document()
        doc.findings = [SimpleNamespace(label="IP", role="primary"), SimpleNamespace(label="ACTION", role="primary")]
        doc.review_findings = [{"kind": "extraction"}, {"kind": "extraction"}]
        tasks = summarize_tasks(doc, "ner", ioc_checked=True)
        self.assertEqual(tasks["message"]["finding_count"], 0)
        self.assertEqual(tasks["message"]["status"], "no_pattern_detected")
        self.assertEqual(tasks["extraction"]["status"], "complete")
        self.assertFalse(tasks["extraction"]["presence_is_threat_verdict"])

    def test_zero_extraction_is_completion_not_a_safe_prediction(self):
        tasks = summarize_tasks(document(), "ner", ioc_checked=True)
        self.assertEqual(tasks["extraction"]["status"], "complete")
        self.assertEqual(tasks["extraction"]["primary_findings"], 0)
        self.assertIn("not a threat verdict", tasks["extraction"]["summary"])
        self.assertNotIn("probability", tasks)

    def test_ioc_mode_reports_entity_models_not_requested(self):
        doc = document(); doc.metadata.pop("ner")
        tasks = summarize_tasks(doc, "ioc", ioc_checked=True)
        self.assertEqual(tasks["extraction"]["status"], "ioc_only")
        self.assertEqual(tasks["extraction"]["entity_model_status"], "not_run")
        self.assertIn("not requested", tasks["extraction"]["summary"])

    def test_missing_detector_or_ioc_execution_does_not_claim_complete_coverage(self):
        doc = document(); doc.metadata["ner"].pop("action")
        self.assertEqual(summarize_tasks(doc, "ner", ioc_checked=True)["extraction"]["status"], "partial_coverage")
        self.assertEqual(summarize_tasks(document(), "ner", ioc_checked=False)["extraction"]["status"], "partial_coverage")

    def test_hindi_rules_can_flag_while_english_ner_and_sms_are_out_of_scope(self):
        doc = document()
        doc.metadata.update(language="hi", ner={"entities": {"status": "unsupported_language"}, "action": {"status": "unsupported_language"}},
            message_assessment={"status": "suspicious", "summary": "Rule evidence."},
            message_classifier={"status": "not_applicable"})
        tasks = summarize_tasks(doc, "ner", ioc_checked=True)
        self.assertEqual(tasks["message"]["status"], "suspicious")
        self.assertEqual(tasks["extraction"]["status"], "language_limited")
        self.assertEqual(tasks["message"]["trained_model_languages"], ["en"])

    def test_primary_counts_do_not_double_count_host_components(self):
        doc = document()
        doc.findings = [SimpleNamespace(label=label, role=role) for label, role in
                        (("URL", "primary"), ("DOMAIN", "component"), ("THREAT_ACTOR", "primary"), ("ACTION", "primary"))]
        tasks = summarize_tasks(doc, "ner", ioc_checked=True)["extraction"]
        self.assertEqual((tasks["primary_findings"], tasks["indicators"], tasks["entities"], tasks["actions"], tasks["components"]), (3, 1, 1, 1, 1))

    def test_unavailable_message_assessment_is_not_filled_from_extraction(self):
        doc = Document("Synthetic text")
        tasks = summarize_tasks(doc, "ioc", ioc_checked=True)
        self.assertEqual(tasks["message"]["status"], "partial_coverage")
        self.assertIn("unavailable", tasks["message"]["summary"])

    def test_invalid_mode_and_non_boolean_ioc_coverage_are_rejected(self):
        for mode, flag in (("unknown", True), ("ner", 1)):
            with self.assertRaises(ValueError):
                summarize_tasks(document(), mode, ioc_checked=flag)


if __name__ == "__main__":
    unittest.main()
