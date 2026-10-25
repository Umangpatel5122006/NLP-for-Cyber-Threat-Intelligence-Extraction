
"""
MITRE ATT&CK publishes a free, curated list of known threat groups (APT29, Lazarus Group) and known malware and tools (Mimikatz, Cobalt Strike), including their aliases.
Real CTI systems use lists like this (called gazetteers) next to their models.
We’ll download it, extract the names, and later give the CRF a clue: “this word appears in the known-names list.” 
That lets it recognize names it never saw in training. 
This is a statistical-NLP technique, with no LLM involved.

"""
import json
from pathlib import Path

raw = Path("data/external/enterprise-attack.json")
if not raw.exists():
    raise SystemExit(f"File not found: {raw.resolve()}")

with open(raw, encoding="utf-8") as f:
    objects = json.load(f)["objects"]

names = {"THREAT_ACTOR": set(), "MALWARE_TOOL": set()}
for o in objects:
    if o.get("revoked") or o.get("x_mitre_deprecated"):
        continue
    if o["type"] == "intrusion-set":
        label, aliases = "THREAT_ACTOR", [o["name"]] + o.get("aliases", [])
    elif o["type"] in ("malware", "tool"):
        label, aliases = "MALWARE_TOOL", [o["name"]] + o.get("x_mitre_aliases", [])
    else:
        continue
    names[label].update(a for a in aliases if len(a) >= 4)   # skip very short, noisy names

out = {label: sorted(v) for label, v in names.items()}
with open("data/external/gazetteer.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

for label, v in out.items():
    print(f"{label}: {len(v)} names, e.g. {v[:8]}")