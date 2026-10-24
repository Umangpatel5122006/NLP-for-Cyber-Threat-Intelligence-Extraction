import cti.preprocessing.language_detector
from cti.core.base import Document
from cti.core.registry import create

det = create("language_detector")
for t in ["APT29 used Mimikatz", "यह हमला बहुत खतरनाक है", "हा हल्ला खूप धोकादायक आहे", "આ હુમલો ખતરનાક છે"]:
    print(det.process(Document(text=t)).metadata["language"], "<-", t)