"""
the defanger:

security reports are often shared in public, so authors “defang” dangerous links to stop readers 
from clicking them by accident. A real domain evil.example.com is written as evil[.]example[.]com, 
and https:// becomes hxxps://. Humans can read this, but our regex in P3 can’t, so it would miss most indicators. 
The defanger converts these back to the normal form before the IOC extractor runs.


change this      -----> to this
hxxp://, hxxps:// ---->	http://, https://
[.], (.), {.}, [dot] ----->	.
[:]	----->  :
[@], [at]	-----> @

"""



import re

from cti.core.base import BaseModule, Document
from cti.core.registry import register

_RULES = [
    (re.compile(r"\bhxxp(s?)", re.I), r"http\1"),
    (re.compile(r"\[://\]"), "://"),
    (re.compile(r"[\[\(\{]\s*\.\s*[\]\)\}]"), "."),
    (re.compile(r"[\[\(\{]\s*dot\s*[\]\)\}]", re.I), "."),
    (re.compile(r"[\[\(\{]\s*:\s*[\]\)\}]"), ":"),
    (re.compile(r"[\[\(\{]\s*(?:@|at)\s*[\]\)\}]", re.I), "@"),
]


@register("defanger")
class Defanger(BaseModule):
    """Turn defanged indicators (evil[.]com) back into normal form (evil.com)."""
    name = "defanger"

    def __init__(self, **params):
        self.params = params

    def process(self, doc: Document) -> Document:
        doc.metadata.setdefault("text_before_defang", doc.text)   # keep the original
        text = doc.text
        for pattern, replacement in _RULES:
            text = pattern.sub(replacement, text)
        doc.text = text
        return doc