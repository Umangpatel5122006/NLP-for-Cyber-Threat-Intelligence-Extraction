"""Build curated CTI gazetteers from MITRE ATT&CK and optional training split data.

Enforces strict split isolation: terms mined from labelled data are taken
exclusively from the training split (never dev or test).
Removes hand-added terms that leak into dev or test gold spans.
Separates campaigns into their own gazetteer group.
Filters short terms and English stop words.
Writes data/external/gazetteer.json and data/external/gazetteer_provenance.json.
"""

import argparse
import json
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ATTACK = ROOT / "data" / "external" / "enterprise-attack.json"
DEFAULT_OUT = ROOT / "data" / "external" / "gazetteer.json"
DEFAULT_PROVENANCE = ROOT / "data" / "external" / "gazetteer_provenance.json"

SUFFIXES_TO_STRIP = [
    " Group", " Team", " Gang", " Crew", " Syndicate", " APT",
    " Actors", " Actor", " Threat Group", " Operation", " Campaign"
]

# Terms audited and confirmed to leak into dev and/or test gold annotations
LEAKED_HAND_ADDED_TERMS = {
    # Infrastructure / tool leaks into dev/test
    "scilc.exe",           # 14 in dev
    "telegram",            # 5 in test
    "active directory",    # 5 in test
    "plc",                 # 2 in test
    "domain controller",   # 1 in dev, 2 in test
    "github",              # 1 in dev, 2 in test
    "svchost.exe",         # 1 in test
    "c2 server",           # 6 in dev, 5 in test
    "c&c server",          # 2 in dev
    "c2",                  # 1 in dev, 3 in test
    "c&c",                 # 1 in test
    # Generic actor phrases that match dev/test gold annotations
    "threat actor", "threat actors", "the threat actor", "the threat actors",
    "the adversary", "the attackers", "the actor", "the group", "attacker",
    "attackers", "adversaries", "adversary"
}

# Clean curated infrastructure phrases that do not appear in dev or test gold
CLEAN_CURATED_INFRASTRUCTURE = [
    "command and control", "command and control server", "web shell", "webshell",
    "gitlab", "discord", "cmd.exe", "powershell.exe", "wscript.exe", "cscript.exe",
    "rundll32.exe", "certutil.exe", "regsvr32.exe", "mshta.exe", "scada", "hmi",
    "ics", "c2 infrastructure"
]


def clean_term(text: str) -> str:
    """Normalize whitespace and strip enclosing punctuation/quotes."""
    return text.strip().strip("'\"").strip()


def is_valid_term(term: str, min_len: int = 3) -> bool:
    """Check if term meets minimum length and is not an English stop word."""
    if len(term) < min_len:
        return False
    if term.lower() in ENGLISH_STOP_WORDS:
        return False
    if term.lower() in LEAKED_HAND_ADDED_TERMS:
        return False
    return True


def build_gazetteer(
    attack_path: Path,
    output_path: Path,
    provenance_path: Path,
    train_only: bool = True,
    include_train_entities: bool = False,
    min_len: int = 3,
):
    if not attack_path.exists():
        raise FileNotFoundError(f"ATT&CK file not found: {attack_path}")

    with open(attack_path, encoding="utf-8") as f:
        objects = json.load(f)["objects"]

    names = {
        "THREAT_ACTOR": set(),
        "MALWARE_TOOL": set(),
        "INFRASTRUCTURE": set(),
        "CAMPAIGN": set(),
    }
    provenance = {}

    # 1. Extract from MITRE ATT&CK
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
            label = "CAMPAIGN"  # Separated from THREAT_ACTOR
            aliases = [o["name"]] + o.get("aliases", [])
        else:
            continue

        for a in aliases:
            if not isinstance(a, str):
                continue
            term = clean_term(a)
            if is_valid_term(term, min_len=min_len):
                names[label].add(term)
                provenance[term] = f"ATT&CK:{otype}"

                # Base name without generic suffix
                for suf in SUFFIXES_TO_STRIP:
                    if term.lower().endswith(suf.lower()):
                        base = clean_term(term[:-len(suf)])
                        if is_valid_term(base, min_len=min_len):
                            names[label].add(base)
                            if base not in provenance:
                                provenance[base] = f"ATT&CK:{otype}:stripped_suffix"

    # 2. Add audited clean curated infrastructure phrases
    for phrase in CLEAN_CURATED_INFRASTRUCTURE:
        term = clean_term(phrase)
        if is_valid_term(term, min_len=min_len):
            names["INFRASTRUCTURE"].add(term)
            if term not in provenance:
                provenance[term] = "curated:infrastructure"

    # 3. Optional: Mine entity mentions from training split only
    if include_train_entities:
        if not train_only:
            raise ValueError("Mining labelled data is only permitted with train_only=True")

        raw_path = ROOT / "data" / "raw" / "SMAH-CTI.json"
        train_ids_path = ROOT / "data" / "splits" / "train_report_ids.json"
        if raw_path.exists() and train_ids_path.exists():
            with open(raw_path, encoding="utf-8") as f:
                raw_data = json.load(f)
            with open(train_ids_path, encoding="utf-8") as f:
                train_doc_ids = set(json.load(f))

            mapping = {
                "ThreatActor": "THREAT_ACTOR",
                "MalwareTool": "MALWARE_TOOL",
                "Infrastructure_Indicator": "INFRASTRUCTURE",
            }
            train_rows = [r for r in raw_data if r.get("doc_id") in train_doc_ids]
            for row in train_rows:
                for ent in row.get("entities") or []:
                    lbl = ent.get("label")
                    if lbl in mapping:
                        target_layer = mapping[lbl]
                        txt = clean_term(ent.get("text", ""))
                        words = txt.split()
                        # Restrict to compact noun phrases
                        if len(words) <= 4 and len(txt) <= 40 and "\n" not in txt:
                            if is_valid_term(txt, min_len=min_len):
                                names[target_layer].add(txt)
                                if txt not in provenance:
                                    provenance[txt] = f"train_split:{lbl}"

    # Sort outputs deterministically
    out = {label: sorted(terms) for label, terms in names.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=1, ensure_ascii=False)

    print("=== Gazetteer Build Complete ===")
    for label, terms in out.items():
        print(f"  {label}: {len(terms)} terms (e.g. {terms[:5]})")
    print(f"Saved gazetteer to: {output_path}")
    print(f"Saved provenance ({len(provenance)} terms) to: {provenance_path}")
    return out, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-path", type=Path, default=DEFAULT_RAW_ATTACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--provenance-output", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--train-only",
        action="store_true",
        default=True,
        help="Ensure any terms mined from labelled data come strictly from the train split.",
    )
    parser.add_argument(
        "--include-train-entities",
        action="store_true",
        default=False,
        help="Mine compact entity names from the training split.",
    )
    parser.add_argument("--min-len", type=int, default=3, help="Minimum character length.")
    args = parser.parse_args()

    build_gazetteer(
        attack_path=args.attack_path,
        output_path=args.output,
        provenance_path=args.provenance_output,
        train_only=args.train_only,
        include_train_entities=args.include_train_entities,
        min_len=args.min_len,
    )


if __name__ == "__main__":
    main()
