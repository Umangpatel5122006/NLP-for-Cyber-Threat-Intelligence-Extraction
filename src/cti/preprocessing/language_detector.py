import unicodedata

from cti.core.base import BaseModule, Document
from cti.core.registry import register

_SCRIPTS = {
    "devanagari": "DEVANAGARI",
    "gujarati": "GUJARATI",
    "latin": "LATIN",
}
_MARATHI_HINTS = {"आहे", "आणि", "हे", "या", "मध्ये", "केले"}


def _script_of(text):
    counts = {name: 0 for name in _SCRIPTS}
    for ch in text:
        if not ch.isalpha():
            continue
        unicode_name = unicodedata.name(ch, "")
        for script, marker in _SCRIPTS.items():
            if marker in unicode_name:
                counts[script] += 1
                break
    best = max(counts, key=counts.get)
    return best if counts[best] else "unknown"


@register("language_detector")
class LanguageDetector(BaseModule):
    """Guess the language and store it in doc.metadata['language']."""
    name = "language_detector"

    def __init__(self, **params):
        self.params = params

    def process(self, doc: Document) -> Document:
        if not isinstance(doc.text, str):
            raise TypeError("Document.text must be a string.")
        script = _script_of(doc.text)
        if script == "latin":
            lang = "en"
        elif script == "gujarati":
            lang = "gu"
        elif script == "devanagari":
            words = {
                "".join(
                    ch
                    for ch in word
                    if unicodedata.category(ch)[0] in ("L", "M")
                )
                for word in doc.text.casefold().split()
            }
            lang = "mr" if words & _MARATHI_HINTS else "hi"
        else:
            lang = "unknown"
        doc.metadata["language"] = lang
        doc.metadata["script"] = script
        return doc