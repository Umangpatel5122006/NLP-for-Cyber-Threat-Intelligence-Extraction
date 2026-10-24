import cti.preprocessing.cleaner
from cti.core.base import Document
from cti.core.registry import create

messy = "  APT29   used\nMimikatz\t and\u200b contacted 185.22.10.5  "
doc = create("cleaner").process(Document(text=messy))
print(repr(doc.text))