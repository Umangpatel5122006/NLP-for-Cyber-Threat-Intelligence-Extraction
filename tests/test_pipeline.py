from cti.core.pipeline import Pipeline

pipe = Pipeline.from_config("configs/config.yaml")
text = "  APT29 used   Mimikatz.  It contacted hxxps://evil[.]example[.]com and 185.22.10[.]5.  "
doc = pipe.run(text)

print("Clean text:", doc.text)
print("Language:", doc.metadata["language"])
for s in doc.metadata["sentences"]:
    print(" -", s["text"])

print("\nIOCs found:")
for e in doc.entities:
    print(" ", e.label, "->", e.text)