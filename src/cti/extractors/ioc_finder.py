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


def _literal_pattern(value, label):
    if label != "URL":
        return f"(?i:{re.escape(value)})"
    scheme, separator, remainder = value.partition("://")
    if not separator:
        return re.escape(value)
    authority = re.match(r"[^/?#]*", remainder).group()
    userinfo, at_sign, host_port = authority.rpartition("@")
    prefix = re.escape(userinfo + at_sign) if at_sign else ""
    if not at_sign:
        host_port = authority
    # The library normalizes host/scheme case. URL credentials, path, query
    # and fragment must still match exactly; retain the actual report text.
    return (
        f"(?i:{re.escape(scheme + separator)})" + prefix
        + f"(?i:{re.escape(host_port)})" + re.escape(remainder[len(authority):])
    )


@register("ioc_finder")
class IOCFinder(BaseModule):
    """IOC extraction using the ioc-finder library."""
    name = "ioc_finder"

    def __init__(self, **params):
        self.params = params

    def process(self, doc: Document) -> Document:
        doc.validate_text_tracking()
        found = find_iocs(doc.text)
        doc.invalidate_findings()
        seen = set()
        for key, label in _KEY_TO_LABEL.items():
            for value in found.get(key, []):
                pattern = rf"(?<![A-Za-z0-9]){_literal_pattern(value, label)}(?![A-Za-z0-9])"
                for m in re.finditer(pattern, doc.text):
                    span = (m.start(), m.end(), label)
                    if span in seen:
                        continue
                    seen.add(span)
                    doc.entities.append(
                        doc.project_entity(Entity(text=m.group(), label=label, start=m.start(),
                                                  end=m.end(), source=self.name, confidence=1.0))
                    )
        doc.entities.sort(key=lambda e: e.start)
        return doc
