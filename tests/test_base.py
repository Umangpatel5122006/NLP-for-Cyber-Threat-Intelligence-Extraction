from cti.core.base import BaseModule, Document, Entity
from cti.core.registry import create
import cti.extractors.dummy        # reading this file signs the module up


module = create("dummy")          # looked up by name, not by class
doc = module.process(Document(text="APT29 used Mimikatz"))
print(doc.entities)