import cti.preprocessing.defanger
from cti.core.base import Document
from cti.core.registry import create

text = "hxxps://evil[.]example[.]com/a, 185.22.10[.]5, admin[at]evil(.)com"
doc = create("defanger").process(Document(text=text))
print(doc.text)