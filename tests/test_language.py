import unittest
from unittest.mock import patch

from cti.core.base import Document
from cti.core.pipeline import Pipeline
from cti.extractors.message_classifier import MessageSpamClassifier
from cti.preprocessing.language_detector import LanguageDetector, describe_language


class LanguageCoverageTests(unittest.TestCase):
    def test_original_script_guesses_are_explicitly_unverified(self):
        for text, expected in [("APT29 used Mimikatz", "en"), ("यह हमला बहुत खतरनाक है", "hi"),
                               ("हा हल्ला खूप धोकादायक आहे", "mr"), ("આ હુમલો ખતરનાક છે", "gu")]:
            with self.subTest(language=expected):
                doc = LanguageDetector().process(Document(text))
                self.assertEqual(doc.metadata["language"], expected)
                self.assertFalse(doc.metadata["language_detection"]["verified"])
                self.assertEqual(doc.original_text, text)

    def test_unrecognized_non_latin_scripts_are_not_silently_english(self):
        for text in ("您的账户已冻结，请提供银行密码。", "يرجى إرسال كلمة المرور للحساب المصرفي",
                     "Ваш банковский счёт заблокирован", "은행 계좌 비밀번호를 보내주세요"):
            with self.subTest(text=text):
                details = describe_language(text + " https://account-alert.example/verify")
                self.assertNotEqual(details["language"], "en")
                self.assertFalse(details["verified"])

    def test_ascii_iocs_do_not_override_gujarati_prose(self):
        details = describe_language("આ હુમલો ખતરનાક છે https://very-long-account-check.example/verify user@example.org " + "a" * 64)
        self.assertEqual(details["language"], "gu")
        self.assertNotIn("latin", details["script_counts"])

    def test_mixed_script_coverage_is_uncertain(self):
        details = describe_language("Please reply आपके खाते में राशि जमा करने के लिए उत्तर दें")
        self.assertEqual(details["script"], "mixed")
        self.assertEqual(details["language"], "hi-en")

    def test_latin_guesses_remain_unverified_with_distinct_hinglish_routing(self):
        for text, status in (("Su cuenta bancaria necesita una nueva clave", "assumed_english"), ("Aapke bank account me paise milenge reply karo", "guessed_hinglish")):
            with self.subTest(text=text):
                details = describe_language(text)
                self.assertEqual(details["status"], status)
                self.assertFalse(details["verified"])
                self.assertIn("not verified", details["note"])

    def test_indicator_only_and_nonlinguistic_text_are_undetermined(self):
        for text in ("1234 \U0001f6e1", "https://example.org/reward", "a" * 64):
            with self.subTest(text=text):
                self.assertEqual(describe_language(text)["status"], "undetermined")

    def test_non_english_negative_pattern_result_is_not_a_supported_negative(self):
        doc = Pipeline.from_config("configs/config.yaml").run("आपके खाते में राशि जमा होगी।")
        self.assertEqual(doc.metadata["message_risk"]["status"], "no_pattern_detected")
        self.assertEqual(doc.metadata["message_assessment"]["status"], "partial_coverage")
        self.assertEqual(doc.review_findings, [])

    def test_positive_english_cues_in_mixed_text_are_preserved_with_partial_coverage(self):
        text = "यह एक संदेश है कृपया इसकी जांच करें सावधानी रखें अपनी जानकारी सुरक्षित रखें। Claim cash prize. Reply now."
        doc = Pipeline.from_config("configs/config.yaml").run(text)
        self.assertNotEqual(doc.metadata["language"], "en")
        self.assertEqual(doc.metadata["message_risk"]["status"], "suspicious")
        self.assertEqual(doc.metadata["message_risk"]["coverage"]["status"], "applied")
        self.assertTrue(any(r["kind"] == "message_rule" for r in doc.review_findings))

    def test_unsupported_language_does_not_load_or_predict_sms_model(self):
        doc = LanguageDetector().process(Document("આ હુમલો ખતરનાક છે"))
        module = MessageSpamClassifier("does-not-exist.joblib")
        with patch.object(module, "readiness", side_effect=AssertionError("Do not load model for unsupported language")):
            module.process(doc)
        self.assertEqual(doc.metadata["message_classifier"]["status"], "not_applicable")

    def test_iocs_remain_source_aligned_in_non_english_text(self):
        text = "\ufeff\U0001f6e1 您的账户已冻结。https://bank-check[.]example[.]com/verify 185.22.10[.]5"
        doc = Pipeline.from_config("configs/config.yaml").run(text)
        self.assertNotEqual(doc.metadata["language"], "en")
        self.assertTrue(any(r.label == "IP" for r in doc.findings))
        self.assertTrue(any(r.label == "URL" for r in doc.findings))
        for finding in doc.findings:
            self.assertEqual(text[finding.original_start:finding.original_end], finding.original_text)


if __name__ == "__main__":
    unittest.main()
