"""
MITRE ATT&CK publishes a free, curated list of known threat groups (APT29, Lazarus Group) and known malware and tools (Mimikatz, Cobalt Strike), including their aliases.
Real CTI systems use lists like this (called gazetteers) next to their models.
We extract the names and aliases, normalize suffix variations (e.g. 'Lazarus Group' -> 'Lazarus'), and extract infrastructure indicators.
That lets statistical models (CRF) recognize names and indicators never seen in training.
"""
import json
from pathlib import Path

raw = Path("data/external/enterprise-attack.json")
if not raw.exists():
    raise SystemExit(f"File not found: {raw.resolve()}")

with open(raw, encoding="utf-8") as f:
    objects = json.load(f)["objects"]

names = {
    "THREAT_ACTOR": set(),
    "MALWARE_TOOL": set(),
    "INFRASTRUCTURE": set(),
}

SUFFIXES_TO_STRIP = [
    " Group", " Team", " Gang", " Crew", " Syndicate", " APT",
    " Actors", " Actor", " Threat Group", " Operation", " Campaign"
]

for o in objects:
    if o.get("revoked") or o.get("x_mitre_deprecated"):
        continue
    otype = o.get("type")
    if otype == "intrusion-set":
        label = "THREAT_ACTOR"
        aliases = [o["name"]] + o.get("aliases", [])
    elif otype in ("malware", "tool"):
        label = "MALWARE_TOOL"
        aliases = [o["name"]] + o.get("x_mitre_aliases", [])
    elif otype == "campaign":
        label = "THREAT_ACTOR"
        aliases = [o["name"]] + o.get("aliases", [])
    else:
        continue

    for a in aliases:
        if not isinstance(a, str):
            continue
        clean_a = a.strip()
        if len(clean_a) >= 3:
            names[label].add(clean_a)
            # Add base name without generic group/team suffix
            for suf in SUFFIXES_TO_STRIP:
                if clean_a.lower().endswith(suf.lower()) and len(clean_a) - len(suf) >= 3:
                    names[label].add(clean_a[:-len(suf)].strip())

# Common generic actor, tool, and infrastructure phrases in CTI reports
for actor_phrase in [
    "threat actor", "threat actors", "the threat actor", "the threat actors",
    "the adversary", "the attackers", "the actor", "the group", "attacker",
    "attackers", "adversaries", "adversary"
]:
    names["THREAT_ACTOR"].add(actor_phrase)

for infra_phrase in [
    "c2 server", "c&c server", "c2 servers", "c&c servers", "c2", "c&c",
    "command and control", "command and control server", "web shell", "webshell",
    "active directory", "telegram", "github", "gitlab", "discord",
    "cmd.exe", "powershell.exe", "wscript.exe", "cscript.exe", "rundll32.exe",
    "svchost.exe", "scilc.exe", "certutil.exe", "regsvr32.exe", "mshta.exe",
    "scada", "plc", "hmi", "ics", "domain controller", "c2 infrastructure"
]:
    names["INFRASTRUCTURE"].add(infra_phrase)

out = {label: sorted(v) for label, v in names.items()}
with open("data/external/gazetteer.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

for label, v in out.items():
    print(f"{label}: {len(v)} names, e.g. {v[:8]}")
