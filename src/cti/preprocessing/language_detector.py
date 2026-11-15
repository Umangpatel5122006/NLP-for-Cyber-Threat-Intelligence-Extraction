import re
import unicodedata
from collections import Counter

from cti.core.base import BaseModule, Document
from cti.core.registry import register

_SCRIPTS = {
    "devanagari": "DEVANAGARI",
    "gujarati": "GUJARATI",
    "latin": "LATIN",
    "arabic": "ARABIC", "cyrillic": "CYRILLIC", "han": "CJK",
    "hiragana": "HIRAGANA", "katakana": "KATAKANA", "hangul": "HANGUL",
    "bengali": "BENGALI", "tamil": "TAMIL", "telugu": "TELUGU",
    "kannada": "KANNADA", "malayalam": "MALAYALAM", "gurmukhi": "GURMUKHI",
    "odia": "ORIYA", "thai": "THAI", "greek": "GREEK", "hebrew": "HEBREW",
}
_MARATHI_HINTS = {"आहे", "आणि", "हे", "या", "मध्ये", "केले"}
# Count prose rather than ASCII hosts/hashes embedded in another script. This
# affects only the guess; original text and IOC extraction remain unchanged.
_INDICATORS = re.compile(
    r"(?:h(?:tt|xx)ps?://|www\.)\S+|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|"
    r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?|\b[a-f0-9]{32,128}\b", re.I)


def _script_of(text):
    return describe_language(text)["script"]


def describe_language(text):
    """A disclosed script heuristic, not a trained language identifier."""
    counts = Counter()
    for ch in _INDICATORS.sub(" ", text):
        if not ch.isalpha():
            continue
        unicode_name = unicodedata.name(ch, "")
        for script, marker in _SCRIPTS.items():
            if marker in unicode_name:
                counts[script] += 1
                break
        else:
            counts["other"] += 1
    best = max(counts, key=counts.get) if counts else "unknown"
    script = best if not counts or counts[best] / sum(counts.values()) >= 0.8 else "mixed"
    lang = "unknown"
    if script == "latin":
        lang, status = "en", "assumed_english"
        note = "English is assumed from Latin script, not verified. Other Latin-script languages and Hinglish are not validated."
    else:
        if script == "gujarati":
            lang = "gu"
        elif script == "devanagari":
            words = {"".join(ch for ch in word if unicodedata.category(ch)[0] in ("L", "M"))
                     for word in text.casefold().split()}
            lang = "mr" if words & _MARATHI_HINTS else "hi"
        status = "mixed_script" if script == "mixed" else "unsupported_script" if counts else "undetermined"
        note = "English-only NLP coverage is unavailable or uncertain for this text. IOC patterns can still be extracted; no safety verdict is established."
    return {"language": lang, "script": script, "script_counts": dict(sorted(counts.items())),
            "method": "unicode_script_heuristic_v2", "status": status, "verified": False, "note": note}


@register("language_detector")
class LanguageDetector(BaseModule):
    """Guess the language and store it in doc.metadata['language']."""
    name = "language_detector"

    def __init__(self, **params):
        self.params = params

    def process(self, doc: Document) -> Document:
        if not isinstance(doc.text, str):
            raise TypeError("Document.text must be a string.")
        details = describe_language(doc.text)
        doc.metadata["language"] = details["language"]
        doc.metadata["script"] = details["script"]
        doc.metadata["language_detection"] = details
        return doc
