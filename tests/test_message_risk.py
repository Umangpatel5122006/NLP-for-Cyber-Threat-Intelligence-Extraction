import unittest

from cti.core.pipeline import Pipeline
from cti.extractors.message_risk import assess_message


USER_MESSAGE = (
    "PM Modi is giving 15 lakhs rupees to every citizen of our country . "
    "To credit it in you bank account .\\\nKindly SMS 6969 on following number :\\\n6969696969"
)


class MessageRiskTests(unittest.TestCase):
    def test_reported_money_sms_lure_is_suspicious_without_technical_iocs(self):
        doc = Pipeline.from_config("configs/config.yaml").run(USER_MESSAGE)
        risk = doc.metadata["message_risk"]
        self.assertEqual(risk["status"], "suspicious")
        self.assertIn("money_contact_lure", risk["matched_rules"])
        self.assertEqual(doc.findings, [])
        self.assertIn("6969696969", [s["original_text"] for s in risk["signals"]])
        self.assertNotIn("confidence", risk)

    def test_patterns_generalize_beyond_identity_number_and_line_breaks(self):
        for text in (
            "You have won 8 crore rupees! Credit the prize to your bank account. Reply to +91 98765 43210.",
            "The government is giving ₹50000. Claim the money in your bank account; call 9123456789.",
            "Claim a cash reward in your bank account. Visit hxxps://claim[.]example/reward.",
            USER_MESSAGE.replace("\\\n", "\r\n").replace("6969696969", "9123456789"),
        ):
            with self.subTest(text=text):
                self.assertEqual(assess_message(text)["status"], "suspicious")

    def test_credential_requests_have_source_evidence(self):
        for text in (
            "Please share your OTP at https://verify.example/login.",
            "Send your password to 9876543210 to verify your account.",
            "Reply with your PIN to +91 91234 56789.",
        ):
            with self.subTest(text=text):
                risk = assess_message(text)
                self.assertEqual(risk["status"], "suspicious")
                self.assertIn("credential", {s["kind"] for s in risk["signals"]})

    def test_advance_fee_lure(self):
        result = assess_message("You won a cash prize. Pay a processing fee to claim your reward.")
        self.assertIn("advance_fee_lure", result["matched_rules"])

    def test_benign_notifications_and_isolated_cues_are_not_threat_verdicts(self):
        for text in (
            "Your bank account was credited with INR 1000. No action is required.",
            "SMS 6969 to 6969696969 for the bus timetable.",
            "The prime minister discussed banking policy.",
            "My salary is 15 lakh rupees and my bank account is active.",
            "Never share your OTP. Call 9876543210 for support.",
            "The government is not giving 15 lakhs rupees. SMS 9876543210 for news.",
            "Send the meeting notes to the team tomorrow.",
        ):
            with self.subTest(text=text):
                result = assess_message(text)
                self.assertEqual(result["status"], "no_pattern_detected")
                self.assertIn("does not establish", result["summary"])

    def test_warning_example_context_is_distinguished_from_a_direct_lure(self):
        result = assess_message("Beware of this fake message: " + USER_MESSAGE)
        self.assertEqual(result["status"], "review_context")
        self.assertIn("warning", {s["kind"] for s in result["signals"]})

    def test_appended_warning_does_not_hide_prior_suspicious_request(self):
        result = assess_message(USER_MESSAGE + " Never share your password with anyone.")
        self.assertEqual(result["status"], "suspicious")

    def test_distant_unrelated_paragraphs_cannot_combine(self):
        result = assess_message("You won 15 lakh rupees in a prize. " + "Background. " * 100 + "SMS 9876543210 about your bank account.")
        self.assertEqual(result["status"], "no_pattern_detected")

    def test_source_offsets_preserve_bom_emoji_crlf_and_exact_text(self):
        text = "\ufeff\U0001f6e1\r\n" + USER_MESSAGE.replace("\\\n", "\r\n")
        result = assess_message(text)
        self.assertEqual(result["status"], "suspicious")
        for signal in result["signals"]:
            self.assertEqual(text[signal["original_start"]:signal["original_end"]], signal["original_text"])
        self.assertEqual(result["offset_unit"], "unicode_code_points")

    def test_reused_pipeline_replaces_risk_without_leaking_prior_message(self):
        pipeline = Pipeline.from_config("configs/config.yaml")
        self.assertEqual(pipeline.run(USER_MESSAGE).metadata["message_risk"]["status"], "suspicious")
        self.assertEqual(pipeline.run("Meeting at noon.").metadata["message_risk"]["signals"], [])


if __name__ == "__main__":
    unittest.main()
