import cti.extractors.ioc_regex
from cti.core.base import Document
from cti.core.registry import create

text = ("APT29 exploited CVE-2021-44228 and contacted 185.22.10.5 and 999.1.1.1. "
        "Dropper hash: d41d8cd98f00b204e9800998ecf8427e.")
doc = create("ioc_regex").process(Document(text=text))
for e in doc.entities:
    print(e.label, "->", e.text)