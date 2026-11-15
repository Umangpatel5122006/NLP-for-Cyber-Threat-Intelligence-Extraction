import unittest
from itertools import product

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

    def test_phone_deletion_shortening_and_masking_do_not_erase_request_risk(self):
        prefix = USER_MESSAGE.rsplit("6969696969", 1)[0]
        for target in ("6969696969", "696969696", "69696969", "6969", "6", "", "69xxxx696", "+91 69696 96969"):
            with self.subTest(target=target):
                text = prefix + target
                result = assess_message(text)
                self.assertEqual(result["status"], "suspicious")
                self.assertIn("money_contact_lure", result["matched_rules"])
                self.assertEqual(result["rule_version"], 2)
                for signal in result["signals"]:
                    self.assertEqual(text[signal["original_start"]:signal["original_end"]], signal["original_text"])
        absent = assess_message(prefix)
        self.assertNotIn("phone", {s["kind"] for s in absent["signals"]})

    def test_money_request_variations_use_intent_cues_not_one_name_or_number(self):
        names = ("The government", "A benefit scheme", "A lottery office", "An unknown sender")
        amounts = ("15 lakhs rupees", "Rs.1500000", "INR 1500000", "\u20b91500000", "cash", "money")
        accounts = ("bank account", "account details")
        requests = ("Kindly SMS 6969 on following number:", "Please reply to claim the payment.",
                    "To receive the funds, text the confirmation code.", "Call to get the credit.")
        targets = ("6969696969", "696969696", "69xx696", "")
        for name, amount, account, request, target in product(names, amounts, accounts, requests, targets):
            text = f"{name} is giving {amount} to everyone. To credit it in your {account}.\n{request}\n{target}"
            with self.subTest(text=text):
                self.assertEqual(assess_message(text)["status"], "suspicious")

    def test_currency_amount_removal_and_whitespace_changes_preserve_money_lure(self):
        for text in (
            "You can claim money in your bank account. Please reply.",
            "A grant is giving lakhs rupees for your bank account. Kindly text us.",
            "Claim a cash reward in your bank account by replying now.",
            "\ufeff\U0001f6e1 You will receive cash in your bank account.\r\nPLEASE\tSMS 6969.\r\n",
        ):
            with self.subTest(text=text):
                self.assertEqual(assess_message(text)["status"], "suspicious")

    def test_completed_notifications_and_passive_sms_mentions_remain_unflagged(self):
        for text in (
            "Cashback of Rs.50 has been credited to your bank account. Call 9123456789 for support.",
            "Your reward of 500 rupees was credited to your bank account. Please contact support if needed.",
            "We have credited a reward of Rs.50 to your bank account. SMS HELP for support.",
            "We receive SMS confirmations after a 1500 rupees payment into the bank account.",
            "We send SMS after a cash reward is credited to the bank account.",
            "Your statement shows INR 1500 in your bank account. SMS BAL to view your balance.",
            "Enter your OTP in your own banking app to complete the login.",
            "Never share your OTP. Contact support if you need help.",
            "The government is not giving 15 lakhs rupees. SMS NEWS for updates.",
            "You have not won a lottery prize. SMS NEWS for updates.",
        ):
            with self.subTest(text=text):
                self.assertEqual(assess_message(text)["status"], "no_pattern_detected")

    def test_sensitive_disclosure_requests_do_not_require_complete_contact_targets(self):
        for text in ("Please share your OTP.", "Reply with your bank password.", "Text your PIN to this number: 1234."):
            with self.subTest(text=text):
                self.assertEqual(assess_message(text)["status"], "suspicious")

    def test_warning_and_negation_remain_effective_without_phone(self):
        text = "Claim 15 lakhs rupees in your bank account. Kindly SMS 6969."
        self.assertEqual(assess_message("Beware of this fake message: " + text)["status"], "review_context")
        self.assertEqual(assess_message("Never share your OTP.")["status"], "no_pattern_detected")
        self.assertEqual(assess_message("The scheme is not giving 15 lakhs rupees to bank accounts. Please SMS NEWS.")["status"], "no_pattern_detected")

    def test_independent_offer_patterns_survive_removal_of_bank_words(self):
        for text in (
            "The government is giving 15 lakhs rupees to every citizen. Kindly SMS 6969.",
            "You have won a 5000 rupees lottery prize. Please reply to collect it.",
            "An official grant is giving cash to everyone. Call to receive it.",
        ):
            with self.subTest(text=text):
                self.assertEqual(assess_message(text)["status"], "suspicious")


if __name__ == "__main__":
    unittest.main()
