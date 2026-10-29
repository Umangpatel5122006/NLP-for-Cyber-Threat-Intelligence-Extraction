import re

from cti.core.base import BaseModule, Document
from cti.core.registry import register

# a sentence end: . ! ? or the Hindi danda, then optional closing quote/bracket, then space
_BOUNDARY = re.compile(r"[.!?\u0964]+[\"')\]]*\s+")
_ABBREVIATION = re.compile(
    r"(?:^|[\s(])(?:e\.g|i\.e|etc|vs|mr|mrs|dr)$",
    re.IGNORECASE,
)


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
        if not isinstance(text, str):
            raise TypeError("Document.text must be a string.")
        sentences = []
        start = 0
        for m in _BOUNDARY.finditer(text):
            nxt = text[m.end():m.end() + 1]
            prefix = text[:m.start()].rstrip()
            if (
                (nxt and nxt.islower())
                or _ABBREVIATION.search(prefix)
            ):
                continue
            _add(text, start, m.end(), sentences)
            start = m.end()
        _add(text, start, len(text), sentences)
        doc.metadata["sentences"] = sentences
        return doc