from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cti.core.offsets import MappedText


@dataclass
class Entity: 
    """take word from sentence and attach properties of it on below factors"""

    text: str            # the found text, e.g. "CVE-2024-1234"
    label: str           # its type, e.g. "CVE"
    start: int           # start position idx in the document
    end: int             # end position idx in the document
    source: str = ""     # which module(Ex: CRF NER , ioc_regex etc) found it
    confidence: float | None = 1.0  # None when the module has no confidence estimate
    original_start: int | None = None
    original_end: int | None = None
    original_text: str | None = None


@dataclass
class Finding:
    id: str
    text: str
    label: str
    start: int
    end: int
    canonical_value: str | None
    canonical_status: str
    original_start: int
    original_end: int
    original_text: str
    sources: list[str]
    observed_labels: list[str]
    evidence: list[Entity]
    role: str = "primary"
    parent_ids: list[str] = field(default_factory=list)
    component_ids: list[str] = field(default_factory=list)


@dataclass
class FindingOverlap:
    left_id: str
    right_id: str
    kind: str


@dataclass
class Document:
    text: str
    entities: list[Entity] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    overlaps: list[FindingOverlap] = field(default_factory=list)
    review_findings: list[dict] = field(default_factory=list)
    original_text: str = field(init=False)

    def __post_init__(self):
        self.original_text = self.text
        # Internal runtime state is not a dataclass field: asdict/JSON should
        # include evidence spans, not one mapping entry per character.
        self._text_map = MappedText.from_text(self.text)

    def validate_text_tracking(self):
        if self.text != self._text_map.text or self.original_text != self._text_map.original:
            raise ValueError("Document text changed without offset tracking; use transform_text before extraction.")

    def transform_text(self, transform):
        self.validate_text_tracking()
        updated = transform(self._text_map)
        if not isinstance(updated, MappedText) or updated.original != self.original_text:
            raise ValueError("A text transform must preserve the supplied original report.")
        if updated.text != self.text and (self.entities or self.metadata.get("sentences")):
            raise ValueError("Run text normalization before sentence splitting and entity extraction.")
        if updated.text != self.text:
            self.invalidate_findings()
        self.text, self._text_map = updated.text, updated

    def invalidate_findings(self):
        """Derived merge output must be rebuilt after detector/text changes."""
        self.findings.clear()
        self.overlaps.clear()
        self.review_findings.clear()
        self.metadata.pop("review", None)
        self.metadata.pop("message_assessment", None)
        self.metadata.pop("merge", None)

    def original_span(self, start, end):
        self.validate_text_tracking()
        return self._text_map.original_span(start, end)

    def _original_fields(self, text, start, end):
        original_start, original_end = self.original_span(start, end)
        if self.text[start:end] != text:
            raise ValueError("Evidence text must match its document offsets.")
        return {
            "original_start": original_start, "original_end": original_end,
            "original_text": self.original_text[original_start:original_end],
        }

    def project_entity(self, entity):
        fields = self._original_fields(entity.text, entity.start, entity.end)
        for key, value in fields.items():
            setattr(entity, key, value)
        return entity

    def attach_original_offsets(self):
        """Validate all evidence before adding original-report fields."""
        self.validate_text_tracking()
        entity_fields = [self._original_fields(e.text, e.start, e.end) for e in self.entities]
        sentences = self.metadata.get("sentences", [])
        if not isinstance(sentences, list) or any(not isinstance(s, dict) for s in sentences):
            raise ValueError("sentences must be a list of sentence mappings.")
        sentence_fields = [self._original_fields(s.get("text"), s.get("start"), s.get("end")) for s in sentences]
        for entity, fields in zip(self.entities, entity_fields):
            for key, value in fields.items():
                setattr(entity, key, value)
        for sentence, fields in zip(sentences, sentence_fields):
            sentence.update(fields)
        self.metadata["offsets"] = {
            "schema_version": 1, "unit": "unicode_code_points", "interval": "half_open",
            "normalized_text_field": "text", "original_text_field": "original_text",
            "source_span_policy": "covering",
        }
        return self


class BaseModule(ABC):
    """Every pipeline module must follow this rule."""
    name = "base"

    @abstractmethod
    def process(self, doc: Document) -> Document:
        ...
