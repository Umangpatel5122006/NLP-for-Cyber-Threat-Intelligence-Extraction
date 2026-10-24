from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Entity: 
    """take word from sentence and attach properties of it on below factors"""

    text: str            # the found text, e.g. "CVE-2024-1234"
    label: str           # its type, e.g. "CVE"
    start: int           # start position idx in the document
    end: int             # end position idx in the document
    source: str = ""     # which module(Ex: CRF NER , ioc_regex etc) found it
    confidence: float = 1.0  # how sure it is correct NER(Name Entity Recognistion)


@dataclass
class Document:
    text: str
    entities: list[Entity] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseModule(ABC):
    """Every pipeline module must follow this rule."""
    name = "base"

    @abstractmethod
    def process(self, doc: Document) -> Document:
        ...