from cti.core.base import BaseModule, Document
from cti.core.registry import register

_SCRIPTS = {                       # Unicode ranges
    "devanagari": (0x0900, 0x097F),
    "gujarati": (0x0A80, 0x0AFF),
    "latin": (0x0041, 0x007A),
}
_MARATHI_HINTS = {"आहे", "आणि", "हे", "या", "मध्ये", "केले"}


def _script_of(text):
    counts = {name: 0 for name in _SCRIPTS}
    for ch in text:
        for name, (lo, hi) in _SCRIPTS.items():
            if lo <= ord(ch) <= hi:
                counts[name] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] else "unknown"


@register("language_detector")
class LanguageDetector(BaseModule):
    """Guess the language and store it in doc.metadata['language']."""
    name = "language_detector"

    def __init__(self, **params):
        self.params = params

    def process(self, doc: Document) -> Document:
        script = _script_of(doc.text)
        if script == "latin":
            lang = "en"
        elif script == "gujarati":
            lang = "gu"
        elif script == "devanagari":
            words = set(doc.text.split())
            lang = "mr" if words & _MARATHI_HINTS else "hi"
        else:
            lang = "unknown"
        doc.metadata["language"] = lang
        doc.metadata["script"] = script
        return doc