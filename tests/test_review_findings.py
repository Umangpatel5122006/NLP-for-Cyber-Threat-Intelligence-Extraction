import copy
import unittest

from cti.core.base import Document
from cti.core.pipeline import Pipeline
from cti.extractors.message_risk import MessageRisk
from cti.output.merger import ResultMerger
from test_message_risk import USER_MESSAGE


class ReviewFindingsTests(unittest.TestCase):
    def test_scam_has_review_finding_without_inventing_entities_or_iocs(self):
        doc = Pipeline.from_config("configs/config.yaml").run(USER_MESSAGE)
        self.assertEqual(doc.findings, [])
        self.assertEqual(doc.entities, [])
        self.assertEqual(len(doc.review_findings), 1)
        finding = doc.review_findings[0]
        self.assertEqual(finding["kind"], "message_rule")
        self.assertEqual(finding["sources"], ["message_risk"])
        self.assertEqual(len(finding["evidence"]), 7)
        for row in [finding, *finding["evidence"]]:
            self.assertEqual(doc.original_text[row["original_start"]:row["original_end"]], row["original_text"])

    def test_extractions_and_message_finding_coexist_with_stable_ids(self):
        doc = Pipeline.from_config("configs/config.yaml").run(USER_MESSAGE + " Contact 185.22.10[.]5.")
        self.assertTrue(doc.findings)
        self.assertEqual(len(doc.review_findings), len(doc.findings) + 1)
        self.assertEqual({r["extraction_id"] for r in doc.review_findings if r["kind"] == "extraction"}, {r.id for r in doc.findings})
        before = copy.deepcopy(doc.review_findings)
        ResultMerger().process(doc)
        self.assertEqual(doc.review_findings, before)

    def test_model_disagreement_keeps_rule_finding_and_explicit_assessment(self):
        doc = MessageRisk().process(Document(USER_MESSAGE))
        doc.metadata["message_classifier"] = {"status": "ok", "label": "ham"}
        ResultMerger().process(doc)
        self.assertEqual(doc.metadata["message_assessment"]["status"], "suspicious")
        self.assertTrue(doc.metadata["message_assessment"]["disagreement"])
        self.assertEqual(doc.review_findings[0]["kind"], "message_rule")

    def test_positive_model_creates_review_row_without_risk_rules(self):
        doc = Document("A synthetic message for review.")
        doc.metadata["message_classifier"] = {"status": "ok", "label": "spam", "decision_margin": 0.8,
            "supporting_features": [{"feature": "word__prize", "contribution": 0.2}], "profile": "word_char_svm", "limitations": "Not verified fraud."}
        ResultMerger().process(doc)
        self.assertEqual(doc.review_findings[0]["kind"], "message_model")
        self.assertEqual(doc.metadata["message_assessment"]["status"], "suspicious")
        self.assertEqual(doc.findings, [])

    def test_invalid_risk_evidence_rejected_before_derived_outputs_mutate(self):
        doc = Pipeline.from_config("configs/config.yaml").run(USER_MESSAGE)
        before = copy.deepcopy(doc.review_findings)
        doc.metadata["message_risk"]["signals"][0]["original_text"] = "wrong"
        with self.assertRaises(ValueError):
            ResultMerger().process(doc)
        self.assertEqual(doc.review_findings, before)

    def test_rerun_invalidates_combined_view_and_benign_doc_has_no_leakage(self):
        doc = Pipeline.from_config("configs/config.yaml").run(USER_MESSAGE)
        MessageRisk().process(doc)
        self.assertEqual(doc.review_findings, [])
        self.assertNotIn("message_assessment", doc.metadata)
        benign = Pipeline.from_config("configs/config.yaml").run("Lunch at noon tomorrow.")
        self.assertEqual(benign.review_findings, [])
        self.assertEqual(benign.metadata["review"]["message_count"], 0)


if __name__ == "__main__":
    unittest.main()
