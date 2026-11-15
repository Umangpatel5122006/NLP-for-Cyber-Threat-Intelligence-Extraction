"""Explainable English social-engineering patterns, separate from CTI NER.

No identity fact checking, telephone reputation lookup or trained classifier is
performed. Matches describe suspicious combinations, never an attack probability.
All evidence indexes the untouched input in Unicode code points.
"""

from bisect import bisect_left
import re

from cti.core.base import BaseModule
from cti.core.registry import register


def _pattern(value):
    return re.compile(value, re.IGNORECASE)


PATTERNS = {
    "reward": _pattern(
        r"\b(?:giv(?:e|es|ing)|receiv(?:e|es|ing)|get|getting|claim|won|win|awarded|eligible|"
        r"entitled|free|grant|prize|reward|lottery|cashback)\b"
    ),
    "money": _pattern(
        r"(?:\b\d[\d,.]*\s*(?:lakhs?|lacs?|crores?|rupees?|rs\.?|inr|usd|dollars?)\b|"
        r"(?:₹|\$)\s*\d[\d,.]*|\b(?:cash|money|prize|lottery|funds|grant)\b)"
    ),
    "bank": _pattern(r"\b(?:bank\s+account|account\s+(?:number|details)|credit(?:ed)?|deposit|payment)\b"),
    "contact": _pattern(
        r"\b(?:sms|text|reply|respond|call|contact|whatsapp|click|visit|send|pay|transfer|share|provide|enter|submit|confirm)\b"
    ),
    "phone": _pattern(r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\d[ -]?){9,11}\d(?!\w)"),
    "link": _pattern(r"\b(?:https?://|hxxps?://|www\.)[^\s<>]+"),
    "credential": _pattern(r"\b(?:otp|one[ -]time\s+(?:password|code)|password|pin|cvv|verification\s+code)\b"),
    "disclose": _pattern(r"\b(?:send|share|provide|enter|confirm|submit|tell|reply(?:\s+with)?)\b"),
    "fee": _pattern(r"\b(?:processing\s+fee|advance\s+(?:fee|payment)|registration\s+fee|pay\s+(?:a\s+)?fee)\b"),
    "authority": _pattern(r"\b(?:pm|prime\s+minister|government|president|reserve\s+bank|rbi|official)\b"),
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
    "warning": ("Warning or example context", "The text may be describing or warning about a scam."),
}

RULES = (
    ("money_contact_lure", ("reward", "money", "bank", "contact", "phone")),
    ("money_link_lure", ("reward", "money", "bank", "contact", "link")),
    ("credential_request", ("credential", "disclose", "contact", "phone")),
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


def assess_message(text):
    """Match small local cue combinations, independent of NER/IOC availability."""
    matches = {key: list(pattern.finditer(text)) for key, pattern in PATTERNS.items()}
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
        for key in ("reward", "contact", "disclose"):
            nearby[key] = [m for m in nearby[key] if not _negated(text, m)]
        for rule, required in RULES:
            if not all(nearby[key] for key in required):
                continue
            # Warnings before a solicitation can explain why it is quoted.
            # A warning appended after a request does not erase that request.
            warnings = [m for m in nearby["warning"] if m.start() <= anchor.start()]
            contextual.append(bool(warnings))
            rules.add(rule)
            for key in (*required, "authority", "warning"):
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
        "rule_version": 1, "method": "english_rules", "status": status,
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
        doc.metadata["message_risk"] = assess_message(doc.original_text)
        return doc
