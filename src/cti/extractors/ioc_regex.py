import re

from cti.core.base import BaseModule, Document, Entity
from cti.core.registry import register

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"

PATTERNS = {
    "IP": re.compile(rf"(?<![\d.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\d.]*\d)"),
    "CVE": re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.I),
    "HASH": re.compile(r"\b(?:[a-fA-F0-9]{64}|[a-fA-F0-9]{40}|[a-fA-F0-9]{32})\b"),
}


@register("ioc_regex")
class IOCRegex(BaseModule):
    """Find indicators of compromise with regex rules."""
    name = "ioc_regex"

    def __init__(self, **params):
        self.params = params

    def process(self, doc: Document) -> Document:
        for label, pattern in PATTERNS.items():
            for m in pattern.finditer(doc.text):
                doc.entities.append(
                    Entity(text=m.group(), label=label, start=m.start(),
                           end=m.end(), source=self.name, confidence=1.0)
                )
        doc.entities.sort(key=lambda e: e.start)
        return doc