import cti.preprocessing.sentence_splitter
from cti.core.base import Document
from cti.core.registry import create

splitter = create("sentence_splitter")

text = "APT29 used Mimikatz on host 10.0.0.1. It then contacted evil.example.com, e.g. a C2 server. The malware persisted."
for s in splitter.process(Document(text=text)).metadata["sentences"]:
    print(s)

hindi = "यह एक वाक्य है। यह दूसरा वाक्य है।"
print(len(splitter.process(Document(text=hindi)).metadata["sentences"]), "Hindi sentences")