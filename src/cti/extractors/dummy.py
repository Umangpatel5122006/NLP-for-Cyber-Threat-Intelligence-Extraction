from cti.core.base import BaseModule, Document, Entity
from cti.core.registry import register


@register("dummy")
class DummyModule(BaseModule):
    """Toy module: marks the first word of the text as an entity."""
    name = "dummy"

    def __init__(self, **params):
        self.params = params          # settings from config.yaml arrive here

    def process(self, doc: Document) -> Document:
        first_word = doc.text.split()[0]
        doc.entities.append(
            Entity(text=first_word, label="TEST", start=0, end=len(first_word), source=self.name)
        )
        return doc