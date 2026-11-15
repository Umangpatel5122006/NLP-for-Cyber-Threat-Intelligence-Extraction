"""
It do Unicode Normalization by NFC

and fix invisible char 

In English, a \u200b hiding inside Mimikatz would make it a different word to the computer, even though you can’t see it. 
In Hindi or Gujarati, deleting \u200c or \u200d can change how a word is drawn, or even its spelling.

So the cleaner removes only the junk and leaves anything that Indian scripts depend on.

"""



import re

from cti.core.base import BaseModule, Document
from cti.core.registry import register

_INVISIBLE = re.compile("[\u200b\ufeff]")   # junk only, NOT \u200c / \u200d
_SPACES = re.compile(r"\s+")


@register("cleaner")
class TextCleaner(BaseModule):
    """Fix whitespace and invisible characters. Never changes real words."""
    name = "cleaner"

    def __init__(self, **params):
        self.params = params

    def process(self, doc: Document) -> Document:
        if not isinstance(doc.text, str):
            raise TypeError("Document.text must be a string.")
        doc.transform_text(lambda text: text.normalize_nfc().sub(_INVISIBLE, "").sub(_SPACES, " ").strip())
        return doc
