import re

from ioc_finder import find_iocs

from cti.core.base import BaseModule, Document, Entity
from cti.core.registry import register

# library result key -> our label
_KEY_TO_LABEL = {
    "ipv4s": "IP",
    "ipv6s": "IP",
    "domains": "DOMAIN",
    "urls": "URL",
    "email_addresses": "EMAIL",
    "cves": "CVE",
    "md5s": "HASH",
    "sha1s": "HASH",
    "sha256s": "HASH",
    "sha512s": "HASH",
}


@register("ioc_finder")
class IOCFinder(BaseModule):
    """IOC extraction using the ioc-finder library."""
    name = "ioc_finder"

    def __init__(self, **params):
        self.params = params

    def process(self, doc: Document) -> Document:
        found = find_iocs(doc.text)
        seen = set()
        for key, label in _KEY_TO_LABEL.items():
            for value in found.get(key, []):
                pattern = rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])"
                for m in re.finditer(pattern, doc.text):
                    span = (m.start(), m.end(), label)
                    if span in seen:
                        continue
                    seen.add(span)
                    doc.entities.append(
                        Entity(text=m.group(), label=label, start=m.start(),
                               end=m.end(), source=self.name, confidence=1.0)
                    )
        doc.entities.sort(key=lambda e: e.start)
        return doc