import re

from cti.core.base import BaseModule, Document
from cti.core.registry import register

# a sentence end: . ! ? or the Hindi danda, then optional closing quote/bracket, then space
_BOUNDARY = re.compile(r"[.!?\u0964]+[\"')\]]*\s+")


def _add(text, start, end, out):
    chunk = text[start:end]
    stripped = chunk.strip()
    if stripped:
        s = start + (len(chunk) - len(chunk.lstrip()))
        out.append({"start": s, "end": s + len(stripped), "text": stripped})


@register("sentence_splitter")
class SentenceSplitter(BaseModule):
    """Split text into sentences. Stores them in doc.metadata['sentences']."""
    name = "sentence_splitter"

    def __init__(self, **params):
        self.params = params

    def process(self, doc: Document) -> Document:
        text = doc.text
        sentences = []
        start = 0
        for m in _BOUNDARY.finditer(text):
            nxt = text[m.end():m.end() + 1]
            if nxt and nxt.islower():       # "e.g. a" is not a sentence end
                continue
            _add(text, start, m.end(), sentences)
            start = m.end()
        _add(text, start, len(text), sentences)
        doc.metadata["sentences"] = sentences
        return doc