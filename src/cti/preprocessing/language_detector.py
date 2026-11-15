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
_HINGLISH_HINTS = frozenset({
    "aap", "aapka", "aapke", "aapki", "aapko", "apka", "apke", "apki", "apko",
    "apna", "apne", "apni", "kripya", "kripaya", "karo", "karein", "karen", "kariye",
    "bhejo", "bhejein", "bhejiye", "batao", "bataiye", "batayein", "nahi", "nahin",
    "hain", "hoga", "hogi", "milega", "milenge", "milegi", "kyunki", "liye", "chahiye",
    "tumhara", "tumhare", "tumhe", "hamara", "hamare", "humara", "humare", "sarkar", "sarkaar",
    "hua", "hui", "hue", "gaye", "gaya", "gayi", "raha", "rahi", "rahe", "ghar", "aana", "jana", "kal", "aaj",
})
_HINGLISH_STRONG = frozenset({"kripya", "kripaya", "bhejo", "bhejein", "bhejiye", "batao", "bataiye", "batayein"})
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
    tokens = set(re.findall(r"[a-z]+", _INDICATORS.sub(" ", text).casefold()))
    hints = tokens & _HINGLISH_HINTS
    hindi_words = {"".join(ch for ch in word if unicodedata.category(ch)[0] in ("L", "M"))
                   for word in text.casefold().split()}
    marathi = bool(hindi_words & {"आहे", "आहेत", "आणि", "मध्ये"})
    hindi_mixed = bool(counts.get("devanagari") and counts.get("latin") and set(counts) <= {"devanagari", "latin"} and not marathi)
    if (script == "latin" and (len(hints) >= 2 or hints & _HINGLISH_STRONG)) or (script == "mixed" and hindi_mixed):
        lang, status = "hi-en", "guessed_hinglish"
        note = "Hinglish is guessed from script or lexical cues, not verified. Hindi/Hinglish phrase rules are available; trained SMS and CTI models remain English-only."
    elif script == "latin":
        lang, status = "en", "assumed_english"
        note = "English is assumed from Latin script, not verified. Use the Hindi/Hinglish setting if the automatic guess is wrong."
    else:
        if script == "gujarati":
            lang = "gu"
        elif script == "devanagari":
            lang = "mr" if marathi else "hi"
        status = "mixed_script" if script == "mixed" else "unsupported_script" if counts else "undetermined"
        note = "Hindi phrase rules are available; trained SMS and CTI models remain English-only. The script-based language guess is not verified." if lang == "hi" else "Language coverage is unavailable or uncertain for this text. IOC patterns can still be extracted; no safety verdict is established."
        if lang == "hi":
            status = "guessed_hindi"
    return {"language": lang, "script": script, "script_counts": dict(sorted(counts.items())),
            "method": "script_and_lexical_heuristic_v3", "status": status, "verified": False, "note": note}


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
        hint = doc.metadata.get("language_hint", "auto")
        if hint not in ("auto", "en", "hi", "hi-en"):
            raise ValueError("Choose auto, en, hi or hi-en as the language setting.")
        if hint != "auto":
            details.update(language=hint, status="user_selected", note="Language selected by the user, not automatically verified. English has trained models; Hindi/Hinglish currently use phrase rules only.")
        details["requested"] = hint
        doc.metadata["language"] = details["language"]
        doc.metadata["script"] = details["script"]
        doc.metadata["language_detection"] = details
        return doc
