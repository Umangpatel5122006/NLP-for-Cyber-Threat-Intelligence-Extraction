"""Authored Hindi/Hinglish regression fixtures, not real-world evaluation data."""

import itertools
import unittest
from unittest.mock import patch

from cti.core.pipeline import Pipeline
from cti.extractors.message_classifier import MessageSpamClassifier
from cti.preprocessing.language_detector import describe_language
from cti.web.server import ReviewEngine, ReviewError


CASES = [
    ("hi-grant", "सरकार हर नागरिक को १५ लाख रुपये दे रही है। पैसे खाते में प्राप्त करने के लिए कृपया इस नंबर पर SMS करें।", "suspicious"),
    ("hinglish-grant", "Sarkar aapke khate me 8 lakh rupaye de rahi hai. Paise paane ke liye is number par SMS karein.", "suspicious"),
    ("hi-secret", "अपना ओटीपी और पासवर्ड इस नंबर पर भेजें।", "suspicious"),
    ("hinglish-secret", "Aap apna OTP aur PIN mujhe batao.", "suspicious"),
    ("hi-warning", "सावधान! यह एक फर्जी संदेश का उदाहरण है: आप ५० लाख रुपये जीत गए हैं। इनाम पाने के लिए SMS करें।", "review_context"),
    ("hinglish-warning", "Savdhan, farzi SMS ka example: Aap 50 lakh rupaye jeet gaye hain. Inaam paane ke liye reply karein.", "review_context"),
    ("hi-negation", "अपना ओटीपी किसी को मत बताएं।", "no_pattern_detected"),
    ("hinglish-negation", "Kabhi apna OTP kisi ko share mat karo.", "no_pattern_detected"),
    ("hinglish-completed", "Aapke bank account me Rs5000 credit ho gaye. SMS HELP for details.", "no_pattern_detected"),
    ("hi-completed", "आपके बैंक खाते में ₹५००० जमा हो गए हैं। किसी कार्रवाई की जरूरत नहीं है।", "no_pattern_detected"),
    ("hi-ownapp", "अपना ओटीपी केवल अपने ऐप में दर्ज करें।", "no_pattern_detected"),
    ("hinglish-ownapp", "Apna OTP keval apne app me enter karein.", "no_pattern_detected"),
    ("hi-no-fee", "ओटीपी भेजें, कोई शुल्क नहीं है।", "suspicious"),
    ("hinglish-threat", "Apna OTP bhejo nahi to account band kar denge.", "suspicious"),
    ("hi-friend", "कल घर आ जाना और दूध ले आना।", "no_pattern_detected"),
    ("hinglish-friend", "Kal ghar aa jana aur doodh le aana.", "no_pattern_detected"),
    ("hi-negated-offer", "सरकार पैसे नहीं देगी। कृपया SMS करें।", "no_pattern_detected"),
    ("hi-link", "अपना पासवर्ड इस लिंक पर दर्ज करें: hxxps://bank-check[.]example[.]com/login", "suspicious"),
]


class IndicMessageTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline.from_config("configs/config.yaml")

    def test_authored_threat_benign_warning_and_negation_fixtures(self):
        for name, text, expected in CASES:
            with self.subTest(case=name):
                doc = self.pipeline.run(text)
                self.assertIn(doc.metadata["language"], ("hi", "hi-en"))
                self.assertEqual(doc.metadata["message_risk"]["status"], expected)
                self.assertEqual(doc.metadata["message_risk"]["method"], "hindi_hinglish_lexical_rules")
                for s in doc.metadata["message_risk"]["signals"]:
                    self.assertEqual(text[s["original_start"]:s["original_end"]], s["original_text"])

    def test_amount_target_and_request_variants_do_not_depend_on_a_fixed_example(self):
        for amount, target, action in itertools.product(
            ("15", "९", "200000"), ("9876543210", "1234", ""), ("SMS karein", "reply karein", "message karo")):
            text = f"Sarkar aapke khate me {amount} lakh rupaye de rahi hai. Paise paane ke liye {action}: {target}"
            doc = self.pipeline.run(text)
            self.assertEqual(doc.metadata["message_risk"]["status"], "suspicious", text)
            self.assertTrue(any(row["kind"] == "message_rule" for row in doc.review_findings))
            if len(target) < 10:
                self.assertNotIn("phone", {s["kind"] for s in doc.metadata["message_risk"]["signals"]})
        for action in ("भेजें", "भेजे", "शेयर करें", "शेयर करे"):
            doc = self.pipeline.run("अपना पासवर्ड मुझे " + action + "।")
            self.assertEqual(doc.metadata["message_risk"]["status"], "suspicious")

    def test_hindi_offsets_preserve_crlf_bom_emoji_and_removed_invisible_characters(self):
        text = "\ufeff\U0001f6e1 अपना पास\u200bवर्ड\r\nइस नंबर पर भेजें। 185.22.10[.]5"
        doc = self.pipeline.run(text)
        self.assertEqual(doc.metadata["message_risk"]["status"], "suspicious")
        self.assertEqual(doc.original_text, text)
        for row in doc.review_findings:
            self.assertEqual(text[row["original_start"]:row["original_end"]], row["original_text"])
        self.assertTrue(any("\u200b" in s["original_text"] for s in doc.metadata["message_risk"]["signals"]))

    def test_a_later_contrast_or_appended_warning_does_not_erase_a_request(self):
        for text in ("सावधान, अपना OTP कभी share मत करो। लेकिन अपना ओटीपी मुझे भेजो।",
                     "अपना पासवर्ड मुझे बताओ। सावधान, किसी से शेयर मत करें।"):
            with self.subTest(text=text):
                self.assertEqual(self.pipeline.run(text).metadata["message_risk"]["status"], "suspicious")

    def test_unrelated_far_apart_cues_do_not_combine(self):
        text = "सरकार खाते में ५ लाख रुपये दे रही है। " + "सामान्य जानकारी। " * 80 + "कृपया SMS करें।"
        self.assertEqual(self.pipeline.run(text).metadata["message_risk"]["status"], "no_pattern_detected")

    def test_identity_threat_requests_require_a_destination_and_preserve_warning_context(self):
        examples = [
            ("Aapka bank account block ho jayega. PAN verify karein: secure-check.example.in/kyc", "suspicious"),
            ("आपका बैंक खाता बंद हो जाएगा। पैन सत्यापन करें: customer-check.example.in/verify", "suspicious"),
            ("Aapka Aadhaar suspend ho jayega. Update karein: https://identity-check.example.com/update", "suspicious"),
            ("Savdhan, farzi SMS ka example: Aapka account block ho jayega. KYC verify karein: fake-check.example.in", "review_context"),
            ("Apna PAN update karein: portal.example.in/profile", "no_pattern_detected"),
            ("Aapka account blocked nahi hai. PAN update karein: portal.example.in", "no_pattern_detected"),
            ("आपका खाता बंद नहीं है। पैन अपडेट करें: portal.example.in", "no_pattern_detected"),
            ("Aapka bank account block ho jayega. KYC apne official app me verify karein.", "no_pattern_detected"),
        ]
        for text, status in examples:
            with self.subTest(text=text):
                doc = self.pipeline.run(text)
                self.assertEqual(doc.metadata["message_risk"]["status"], status)

    def test_bare_destinations_do_not_become_fabricated_urls_or_email_fragments(self):
        text = "Aapka account block ho jayega. PAN verify karein: \U0001f6e1 secure-check[.]example[.]in/path"
        doc = self.pipeline.run(text)
        cues = [s for s in doc.metadata["message_risk"]["signals"] if s["kind"] == "link"]
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0]["label"], "Domain-like destination")
        self.assertEqual(text[cues[0]["original_start"]:cues[0]["original_end"]], cues[0]["original_text"])
        for finding in doc.findings:
            self.assertEqual(text[finding.original_start:finding.original_end], finding.original_text)
            if finding.label == "URL":
                # The existing IOC library can label a schemeless path as URL;
                # the merger must keep its unvalidated canonical value empty.
                self.assertIsNone(finding.canonical_value)
        mail = self.pipeline.run("Aapka account blocked hai. PAN verify karein: support@help.example.in")
        self.assertNotIn("link", {s["kind"] for s in mail.metadata["message_risk"]["signals"]})

    def test_a_full_url_is_not_repeated_as_its_contained_bare_host(self):
        doc = self.pipeline.run("Aapka account block ho jayega. KYC verify karein: https://help.example.in/verify")
        links = [s for s in doc.metadata["message_risk"]["signals"] if s["kind"] == "link"]
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["original_text"], "https://help.example.in/verify")

    def test_additional_imperatives_are_language_cues_and_work_without_phone_numbers(self):
        for action in ("form bhariye", "form bharein", "verify karein", "confirm kijiye"):
            text = "Sarkar aapke khate me 20 lakh rupaye de rahi hai. Rashi paane ke liye " + action + "."
            doc = self.pipeline.run(text)
            self.assertEqual(doc.metadata["message_risk"]["status"], "suspicious", text)
            self.assertNotIn("phone", {s["kind"] for s in doc.metadata["message_risk"]["signals"]})
        self.assertEqual(describe_language("Form bhariye.")["language"], "hi-en")

    def test_roman_na_distinguishes_prohibition_from_a_polite_request(self):
        for text, status in (("Apna OTP na bhejo.", "no_pattern_detected"),
                             ("Apna OTP share na karo.", "no_pattern_detected"),
                             ("Apna OTP mujhe bhejo na.", "suspicious")):
            with self.subTest(text=text):
                self.assertEqual(self.pipeline.run(text).metadata["message_risk"]["status"], status)

    def test_hindi_or_hinglish_never_loads_the_english_sms_classifier(self):
        module = MessageSpamClassifier("missing.joblib")
        with patch.object(module, "readiness", side_effect=AssertionError("English model must not run")):
            for _, text, _ in CASES:
                doc = self.pipeline.run(text)
                module.process(doc)
                self.assertEqual(doc.metadata["message_classifier"]["status"], "not_applicable")

    def test_manual_language_hint_is_recorded_and_does_not_leak_to_the_next_report(self):
        text = "Please share your password."
        forced = self.pipeline.run(text, language_hint="hi-en")
        self.assertEqual(forced.metadata["language"], "hi-en")
        self.assertEqual(forced.metadata["language_detection"]["status"], "user_selected")
        self.assertFalse(forced.metadata["language_detection"]["verified"])
        self.assertEqual(self.pipeline.run(text).metadata["language"], "en")

    def test_language_inference_uses_benign_lexical_hints_and_does_not_misread_hindi_ya_as_marathi(self):
        self.assertEqual(describe_language("Kal ghar aa jana aur doodh le aana.")["language"], "hi-en")
        self.assertEqual(describe_language("अपना ओटीपी या पासवर्ड भेजें।")["language"], "hi")
        self.assertEqual(describe_language("हा हल्ला आहे")["language"], "mr")

    def test_api_accepts_a_supported_language_setting_and_rejects_invalid_ones(self):
        engine = ReviewEngine(ioc_only=True)
        result = engine.analyze({"mode": "ioc", "text": "Aap apna OTP mujhe batao.", "language": "hi-en"})
        self.assertEqual(result["document"]["metadata"]["message_risk"]["status"], "suspicious")
        for language in ("fr", None, ["hi"]):
            with self.subTest(language=language), self.assertRaises(ReviewError) as raised:
                engine.analyze({"mode": "ioc", "text": "x", "language": language})
            self.assertEqual(raised.exception.code, "invalid_language")


if __name__ == "__main__":
    unittest.main()
