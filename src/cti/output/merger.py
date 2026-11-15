"""Conservative, occurrence-based consolidation of IOC and NER observations."""

import ipaddress
import math
import re
from dataclasses import replace
from urllib.parse import urlsplit

from cti.core.base import BaseModule, Document, Entity, Finding, FindingOverlap
from cti.core.registry import register


IOC_LABELS = frozenset({"IP", "DOMAIN", "URL", "EMAIL", "HASH", "CVE"})
POLICY = "conservative_occurrence_v1"
_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")


def _ip(value):
    if "%" in value:
        raise ValueError("Scoped addresses require a separate normalization policy.")
    return str(ipaddress.ip_address(value))


def _domain(value):
    # Keep Unicode names as evidence until an explicit IDNA policy exists.
    name = value.removesuffix(".")
    if not name.isascii() or len(name) > 253 or not all(_DNS_LABEL.fullmatch(label) for label in name.split(".")):
        raise ValueError("Unsupported DNS name syntax.")
    return name.lower()


def _host(value):
    try:
        return _ip(value)
    except ValueError:
        return _domain(value)


def _url(value):
    match = _SCHEME.match(value)
    if match is None or any(character.isspace() for character in value):
        raise ValueError("Only absolute authority URLs are normalized.")
    parsed = urlsplit(value)
    authority = parsed.netloc
    if not parsed.hostname or value[match.end():match.end() + len(authority)] != authority:
        raise ValueError("URL authority is not aligned to its source text.")
    # Access validates numeric port syntax and range. Preserve its original text.
    _ = parsed.port
    host_start = authority.rfind("@") + 1
    address = authority[host_start:]
    if address.startswith("["):
        host_end = host_start + address.index("]")
        host_start += 1
        host_value = _ip(authority[host_start:host_end])
        canonical_address = "[" + host_value + "]"
        suffix = authority[host_end + 1:]
        if suffix and not suffix.startswith(":"):
            raise ValueError("Invalid text after a bracketed URL host.")
        prefix = authority[:host_start - 1]
    else:
        host_end = host_start + len(address.split(":", 1)[0])
        host_value = _host(authority[host_start:host_end])
        canonical_address = host_value
        suffix = authority[host_end:]
        prefix = authority[:host_start]
    canonical = value[:match.end()].lower() + prefix + canonical_address + suffix + value[match.end() + len(authority):]
    return canonical, (match.end() + host_start, match.end() + host_end, host_value)


def _email(value):
    local, separator, domain = value.rpartition("@")
    if not separator or not local or any(character.isspace() for character in value):
        raise ValueError("Unsupported email syntax.")
    host_value = _domain(domain)
    return local + "@" + host_value, (len(local) + 1, len(value), host_value)


def _canonical(label, text):
    """Return canonical value, availability and optional host offsets.

    This is syntax normalization, never a verdict that an indicator is malicious
    or exists. Unsupported syntax remains visible with its original evidence.
    """
    if label not in IOC_LABELS:
        return None, "not_applicable", None
    try:
        if label == "IP":
            value, host = _ip(text), None
        elif label == "DOMAIN":
            value, host = _domain(text), None
        elif label == "URL":
            value, host = _url(text)
        elif label == "EMAIL":
            value, host = _email(text)
        elif label == "HASH":
            if len(text) not in (32, 40, 64, 128) or re.fullmatch(r"[0-9A-Fa-f]+", text) is None:
                raise ValueError("Unsupported hash syntax.")
            value, host = text.lower(), None
        else:
            if re.fullmatch(r"CVE-[0-9]{4}-[0-9]{4,}", text, re.I) is None:
                raise ValueError("Unsupported CVE syntax.")
            value, host = text.upper(), None
        return value, "available", host
    except ValueError:
        return None, "unavailable", None


def _evidence_key(entity):
    # Null confidence sorts before numeric confidence, without inventing a score.
    confidence = (entity.confidence is not None, entity.confidence or 0.0)
    return entity.label, entity.source, confidence


def _validated_evidence(doc):
    groups = {}
    for entity in doc.entities:
        if not isinstance(entity, Entity):
            raise TypeError("Merge input must contain Entity observations.")
        if not isinstance(entity.label, str) or not entity.label or not isinstance(entity.source, str):
            raise ValueError("Entity labels and sources must be strings; labels cannot be empty.")
        score = entity.confidence
        if score is not None and (type(score) not in (int, float) or not 0 <= score <= 1 or not math.isfinite(score)):
            raise ValueError("Observation confidence must be null or a finite number between zero and one.")
        evidence = doc.project_entity(replace(entity))
        key = entity.start, entity.end, entity.label
        groups.setdefault(key, {})[_evidence_key(evidence)] = evidence
    return {key: sorted(items.values(), key=_evidence_key) for key, items in groups.items()}


def _overlap_kind(left, right):
    if left.label == "ACTION" or right.label == "ACTION":
        return "action_overlap"
    if (left.start, left.end) == (right.start, right.end):
        return "label_conflict"
    if left.start <= right.start and right.end <= left.end or right.start <= left.start and left.end <= right.end:
        return "contained_overlap"
    return "crossing_overlap"


@register("result_merger")
class ResultMerger(BaseModule):
    """Build a derived view, preserving every distinct source observation."""

    name = "result_merger"

    def process(self, doc: Document) -> Document:
        doc.validate_text_tracking()
        groups = _validated_evidence(doc)
        ioc_keys = {}
        for key in groups:
            if key[2] in IOC_LABELS:
                ioc_keys.setdefault(key[:2], []).append(key)
        absorbed = 0
        for start, end, label in list(groups):
            if label != "INFRASTRUCTURE":
                continue
            targets = ioc_keys.get((start, end), [])
            # A single syntactically supported specific IOC can carry generic
            # infrastructure evidence at the same span. Conflicting IOC types
            # remain separate, with an explicit conflict edge.
            if len(targets) == 1 and _canonical(targets[0][2], doc.text[start:end])[0] is not None:
                groups[targets[0]].extend(groups.pop((start, end, label)))
                groups[targets[0]].sort(key=_evidence_key)
                absorbed += 1

        findings, hosts = [], {}
        for index, ((start, end, label), evidence) in enumerate(sorted(groups.items()), 1):
            text = doc.text[start:end]
            canonical, status, host = _canonical(label, text)
            original = evidence[0]
            finding = Finding(
                id=f"finding-{index:04d}", text=text, label=label, start=start, end=end,
                canonical_value=canonical, canonical_status=status,
                original_start=original.original_start, original_end=original.original_end,
                original_text=original.original_text,
                sources=sorted({item.source for item in evidence}),
                observed_labels=sorted({item.label for item in evidence}), evidence=evidence,
            )
            findings.append(finding)
            if host is not None:
                hosts[finding.id] = (start + host[0], start + host[1], host[2])

        overlaps, active = [], []
        for right in findings:
            active = [left for left in active if left.end > right.start]
            for left in active:
                parent = child = None
                for candidate, component in ((left, right), (right, left)):
                    expected = hosts.get(candidate.id)
                    if (
                        expected is not None and component.label in ("IP", "DOMAIN")
                        and component.canonical_status == "available"
                        and expected == (component.start, component.end, component.canonical_value)
                    ):
                        parent, child = candidate, component
                        break
                if parent is not None:
                    parent.component_ids.append(child.id)
                    child.parent_ids.append(parent.id)
                    child.role = "component"
                    kind = "ioc_host_component"
                else:
                    kind = _overlap_kind(left, right)
                overlaps.append(FindingOverlap(left.id, right.id, kind))
            active.append(right)
        for finding in findings:
            finding.parent_ids.sort()
            finding.component_ids.sort()
        evidence_count = sum(len(finding.evidence) for finding in findings)
        details = {
            "schema_version": 1, "policy": POLICY, "offset_space": "document_text",
            "confidence_policy": "per_source_only", "raw_entity_count": len(doc.entities),
            "evidence_count": evidence_count, "duplicate_evidence_count": len(doc.entities) - evidence_count,
            "finding_count": len(findings),
            "primary_finding_count": sum(finding.role == "primary" for finding in findings),
            "component_finding_count": sum(finding.role == "component" for finding in findings),
            "absorbed_infrastructure_count": absorbed, "overlap_count": len(overlaps),
        }
        # All validation and derived work succeed before changing document output.
        doc.findings[:], doc.overlaps[:] = findings, overlaps
        doc.metadata["merge"] = details
        return doc
