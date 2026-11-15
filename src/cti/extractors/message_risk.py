"""Explainable English social-engineering patterns, separate from CTI NER.

No identity fact checking, telephone reputation lookup or trained classifier is
performed. Matches describe suspicious combinations, never an attack probability.
All evidence indexes the untouched input in Unicode code points.
"""

from bisect import bisect_left
import re

from cti.core.base import BaseModule
from cti.core.registry import register


RULE_VERSION = 2


def _pattern(value):
    return re.compile(value, re.IGNORECASE)


PATTERNS = {
    "reward": _pattern(
        r"\b(?:giv(?:e|es|ing)|receiv(?:e|es|ing)|get|getting|claim|won|win|awarded|eligible|"
        r"entitled|free|grant|prize|reward|lottery|cashback|collect|redeem|unlock)\b"
    ),
    "money": _pattern(
        r"(?:\b\d[\d,.]*\s*(?:(?:lakhs?|lacs?|crores?)(?:\s+rupees?)?|rupees?|rs\.?|inr|usd|dollars?)\b|"
        r"\b(?:inr|rs\.?|rupees?|usd)\s*\d[\d,.]*|(?:₹|\$)\s*\d[\d,.]*|"
        r"\b(?:(?:lakhs?|lacs?|crores?)(?:\s+rupees?)?|rupees?|dollars?|cash|money|prize|lottery|funds|grant)\b)"
    ),
    "bank": _pattern(r"\b(?:bank\s+account|account\s+(?:number|details)|credit(?:ed)?|deposit|payment)\b"),
    "contact": _pattern(
        r"\b(?:sms|text(?:ing)?|repl(?:y|ying)|respond(?:ing)?|call(?:ing)?|contact(?:ing)?|"
        r"whatsapp|click(?:ing)?|visit(?:ing)?|send(?:ing)?|pay|transfer|shar(?:e|ing)|"
        r"provid(?:e|ing)|enter|submit|confirm|forward|message|dm|tell)\b"
    ),
    "phone": _pattern(r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\d[ -]?){9,11}\d(?!\w)"),
    "link": _pattern(r"\b(?:https?://|hxxps?://|www\.)[^\s<>]+"),
    "credential": _pattern(r"\b(?:otp|one[ -]time\s+(?:password|code)|password|pin|cvv|verification\s+code)\b"),
    "disclose": _pattern(r"\b(?:send(?:ing)?|shar(?:e|ing)|provid(?:e|ing)|enter|confirm|submit|tell|forward|sms|text(?:ing)?|message|dm|repl(?:y|ying)(?:\s+with)?)\b"),
    "fee": _pattern(r"\b(?:processing\s+fee|advance\s+(?:fee|payment)|registration\s+fee|pay\s+(?:a\s+)?fee)\b"),
    "authority": _pattern(r"\b(?:pm|prime\s+minister|government|president|reserve\s+bank|rbi|official)\b"),
    "prize": _pattern(r"\b(?:won|winner|winning|lottery|jackpot|prize)\b"),
    "warning": _pattern(
        r"\b(?:(?:do\s+not|don't|never)\s+(?:reply|respond|send|share|click|pay|sms|call)|"
        r"beware|scam\s+(?:alert|example|message)|phishing\s+(?:example|message)|"
        r"(?:example|report|warning)\s+of\s+(?:a\s+)?(?:scam|phishing)|"
        r"(?:fake|fraudulent)\s+(?:message|sms))\b"
    ),
}

SIGNALS = {
    "reward": ("Reward promise", "A reward or benefit is offered."),
    "money": ("Money lure", "The message mentions money, a prize or a cash amount."),
    "bank": ("Account or payment context", "The offer refers to an account, credit or payment."),
    "contact": ("Response requested", "The message asks the reader to contact, reply or take action."),
    "phone": ("Contact number", "A phone-like number appears near the request; its reputation is unknown."),
    "link": ("Response link", "A link appears near the request; its reputation is unknown."),
    "credential": ("Sensitive credential", "The message names a secret such as an OTP, password or PIN."),
    "disclose": ("Disclosure requested", "The reader is asked to supply information."),
    "fee": ("Advance fee", "A fee is mentioned alongside a promised benefit."),
    "authority": ("Authority claim", "An authority title is invoked; the claimed identity is unverified."),
    "prize": ("Prize or lottery lure", "A prize, win or lottery is combined with a response request."),
    "warning": ("Warning or example context", "The text may be describing or warning about a scam."),
}

RULES = (
    ("money_contact_lure", ("reward", "money", "bank", "contact")),
    ("money_link_lure", ("reward", "money", "bank", "contact", "link")),
    ("authority_money_request", ("authority", "reward", "money", "contact")),
    ("prize_response_lure", ("prize", "money", "contact")),
    ("credential_request", ("credential", "disclose", "contact")),
    ("credential_link_request", ("credential", "disclose", "contact", "link")),
    ("advance_fee_lure", ("reward", "money", "fee", "contact")),
)


def _negated(text, match):
    """Conservative local guard; this is not a general negation parser."""
    prefix = text[max(0, match.start() - 55):match.start()]
    return bool(re.search(
        r"\b(?:not|never|don't|doesn't|isn't|won't|cannot|can't|no)\b[^.!?;\n]{0,35}$",
        prefix, re.IGNORECASE,
    ))


def _is_request(text, match):
    """Recognize directive contexts rather than treating every SMS mention as a request."""
    prefix = text[max(0, match.start() - 120):match.start()]
    clause = re.split(r"[.!?;\n]", prefix)[-1]
    return bool(
        not clause.strip()
        or re.search(r"\b(?:kindly|please|just|now|then|and|by|via)\s*$", clause, re.I)
        or re.search(r"\byou\s+(?:must|should|can|need\s+to|have\s+to)\s*$", clause, re.I)
        or re.search(r"[,:]\s*$", clause)
    )


def _completed_notice(text, nearby, start, end):
    """Completed credit notices need more than an informational contact line."""
    window = text[start:end]
    completed = re.search(r"\b(?:was|were|has(?:\s+been)?|have(?:\s+been)?|already|is)\s+(?:credited|deposited|paid|transferred)\b", window, re.I)
    future_offer = any(re.fullmatch(r"giving|give|claim|collect|redeem|unlock|eligible|entitled|receive", m.group(), re.I)
                       for m in nearby["reward"])
    return bool(completed and not future_offer)


def assess_message(text):
    """Match small local cue combinations, independent of NER/IOC availability."""
    matches = {key: list(pattern.finditer(text)) for key, pattern in PATTERNS.items()}
    for key in ("reward", "prize", "contact", "disclose"):
        matches[key] = [m for m in matches[key] if not _negated(text, m)]
    matches["contact"] = [m for m in matches["contact"] if _is_request(text, m)]
    positions = {key: [m.start() for m in values] for key, values in matches.items()}
    evidence, rules, contextual = {}, set(), []
    # A request is the anchor. Unrelated cues elsewhere in a long report cannot
    # combine; only the surrounding 600 characters can support a pattern.
    for anchor in matches["contact"]:
        start, end = max(0, anchor.start() - 300), anchor.end() + 300
        nearby = {
            key: [m for m in values[bisect_left(positions[key], start):bisect_left(positions[key], end)] if m.end() <= end]
            for key, values in matches.items()
        }
        for rule, required in RULES:
            if not all(nearby[key] for key in required):
                continue
            if rule in ("money_contact_lure", "money_link_lure", "authority_money_request", "prize_response_lure") and _completed_notice(text, nearby, start, end):
                continue
            if rule == "credential_request" and not any(
                re.fullmatch(r"send(?:ing)?|shar(?:e|ing)|provid(?:e|ing)|tell|forward|sms|text(?:ing)?|message|dm|repl(?:y|ying)(?:\s+with)?", m.group(), re.I)
                for m in nearby["disclose"]
            ):
                # Entering a code in one's own app is not the same as sending
                # a secret to a message sender. Link-specific rules stay separate.
                continue
            # Warnings before a solicitation can explain why it is quoted.
            # A warning appended after a request does not erase that request.
            warnings = [m for m in nearby["warning"] if m.start() <= anchor.start()]
            contextual.append(bool(warnings))
            rules.add(rule)
            for key in dict.fromkeys((*required, "phone", "link", "authority", "warning")):
                for match in nearby[key][:3]:
                    evidence[(key, match.start(), match.end())] = {
                        "kind": key, "label": SIGNALS[key][0], "explanation": SIGNALS[key][1],
                        "original_start": match.start(), "original_end": match.end(),
                        "original_text": match.group(),
                    }
    if not rules:
        status = "no_pattern_detected"
        summary = "No supported scam pattern detected. This does not establish that the message is safe."
    elif all(contextual):
        status = "review_context"
        summary = "Scam-related patterns appear in warning or example context. Review the original text."
    else:
        status = "suspicious"
        summary = "Possible social engineering: suspicious request patterns detected. Verify the sender independently before responding."
    return {
        "rule_version": RULE_VERSION, "method": "english_rules", "status": status,
        "summary": summary, "matched_rules": sorted(rules),
        "signals": sorted(evidence.values(), key=lambda item: (item["original_start"], item["kind"])),
        "limitations": "English rule patterns only; no fact checking, reputation lookup or calibrated threat probability. Missing a pattern is not a safety verdict.",
        "offset_unit": "unicode_code_points",
        "offset_interval": "half_open",
    }


@register("message_risk")
class MessageRisk(BaseModule):
    name = "message_risk"

    def process(self, doc):
        doc.invalidate_findings()
        from cti.preprocessing.language_detector import describe_language
        details = doc.metadata.get("language_detection") or describe_language(doc.text)
        language = doc.metadata.get("language", details["language"])
        if language in ("hi", "hi-en"):
            from cti.extractors.indic_message_risk import assess_indic_document
            result = assess_indic_document(doc)
        else:
            result = assess_message(doc.original_text)
        supported = language in ("en", "hi", "hi-en")
        positive = result["status"] in ("suspicious", "review_context")
        result["coverage"] = {"supported_languages": ["en", "hi", "hi-en"], "language": language,
                              "status": "applied" if supported else "partial" if positive else "unsupported_language"}
        if not supported and not positive:
            result.update(status="unsupported_language", summary="English scam-pattern coverage is unavailable or uncertain for this text. No safety conclusion can be made.")
        doc.metadata["message_risk"] = result
        return doc
