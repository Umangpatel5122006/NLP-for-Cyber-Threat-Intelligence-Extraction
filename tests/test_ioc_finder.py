import cti.extractors.ioc_finder
from cti.core.base import Document
from cti.core.registry import create

text = ("APT29 exploited CVE-2021-44228, contacted 185.22.10.5 and "
        "https://evil.example.com/a.php, mailed admin@evil.com. "
        "Hash: d41d8cd98f00b204e9800998ecf8427e.")
doc = create("ioc_finder").process(Document(text=text))
for e in doc.entities:
    print(e.label, "->", e.text, (e.start, e.end))