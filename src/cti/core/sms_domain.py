"""Pinned smishing supplement with source-wide UCI exclusion and grouped splits."""

import csv
import hashlib
import html
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from cti.core.sms_data import DATA, ROOT, read_rows, validate_split


SOURCE = ROOT / "models/domain_research/mendeley_5971.csv"
SPLIT = ROOT / "data/splits/sms_domain_v1.json"
SOURCE_URL = "https://raw.githubusercontent.com/nmbenton/INFO-4360-Project/938fe00b100f88b7b9e4d0b9fb4225161f1e08ed/data/Dataset_5971.csv"
SOURCE_SHA256 = "649844f1c62a6b27e145eaf17a65f7010c56c390e11a794ea8329993a05ba71e"
SIMILARITY = 0.85


def download_domain(path=SOURCE):
    with urlopen(SOURCE_URL, timeout=40) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000 or hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Smishing supplement differs from the reviewed pinned copy.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)


def canonical(text):
    text = html.unescape(unicodedata.normalize("NFKC", text)).casefold()
    text = re.sub(r"(?:https?://|www\.)\S+", " urltoken ", text)
    text = re.sub(r"\b[^\s@]+@[^\s@]+\b", " emailtoken ", text)
    text = re.sub(r"\d+(?:[.,]\d+)*", " numtoken ", text)
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def source_rows(path=SOURCE):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Supplement hash changed.")
    records = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if len(records) != 5971 or Counter(r["LABEL"].strip().casefold() for r in records) != {"ham": 4844, "spam": 489, "smishing": 638}:
        raise ValueError("Supplement schema/class counts differ from the published dataset.")
    return [{"id": f"domain-{i:05d}", "text": html.unescape(r["TEXT"]).strip(),
             "label": "ham" if r["LABEL"].strip().casefold() == "ham" else "spam",
             "original_label": r["LABEL"].strip().casefold(), "source": "mendeley"}
            for i, r in enumerate(records) if r["TEXT"].strip()]


def audit_overlap(rows, protected_texts):
    """Fixed stateless fingerprints only; protected corpus labels are not used."""
    from sklearn.feature_extraction.text import HashingVectorizer

    known = [canonical(t) for t in protected_texts]
    known_set = set(known)
    candidates = [r for r in rows if canonical(r["text"]) and canonical(r["text"]) not in known_set]
    vectorizer = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**18,
                                  alternate_sign=False, norm="l2")
    protected = vectorizer.transform(known)
    matrix = vectorizer.transform([canonical(r["text"]) for r in candidates])
    nearest = []
    for start in range(0, len(candidates), 128):
        nearest.extend((matrix[start:start + 128] @ protected.T).max(axis=1).toarray().ravel().tolist())
    keep = [r for r, similarity in zip(candidates, nearest) if similarity < SIMILARITY]
    return keep, {"input_rows": len(rows), "exact_excluded": len(rows) - len(candidates),
                  "near_excluded": len(candidates) - len(keep), "retained_rows": len(keep),
                  "retained_original_labels": dict(Counter(r["original_label"] for r in keep)),
                  "policy": "canonical exact + stateless char_wb(3,5) hashing cosine >= 0.85",
                  "protected_labels_used": False}


def group_rows(rows):
    from sklearn.feature_extraction.text import HashingVectorizer

    rows = [dict(r) for r in rows]
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    vectorizer = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**18,
                                  alternate_sign=False, norm="l2")
    matrix = vectorizer.transform([canonical(r["text"]) for r in rows])
    edges = (matrix @ matrix.T).tocoo()
    for i, j, value in zip(edges.row, edges.col, edges.data):
        if i < j and value >= SIMILARITY:
            left, right = find(i), find(j)
            parents[max(left, right)] = min(left, right)
    components = {}
    for i, row in enumerate(rows):
        components.setdefault(find(i), []).append(row["id"])
    for i, row in enumerate(rows):
        row["group"] = "domain-" + hashlib.sha256("|".join(sorted(components[find(i)])).encode()).hexdigest()
    return rows


def prepare_domain(source=SOURCE, uci_path=DATA, split_path=SPLIT):
    from sklearn.model_selection import StratifiedGroupKFold

    retained, audit = audit_overlap(source_rows(source), [r["text"] for r in read_rows(uci_path)])
    rows = group_rows(retained)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=73).split(
        rows, [r["label"] for r in rows], [r["group"] for r in rows]))
    test, dev = set(folds[0][1]), set(folds[1][1])
    partitions = {"train": [], "dev": [], "test": []}
    for i, row in enumerate(rows):
        partitions["test" if i in test else "dev" if i in dev else "train"].append(row["id"])
    manifest = {"schema_version": 1, "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "protected_uci_sha256": hashlib.sha256(Path(uci_path).read_bytes()).hexdigest(),
                "protocol": "5 stratified near-template group folds seed73; fold0=test, fold1=dev, rest=train",
                "overlap_audit": audit, "partitions": partitions,
                "group_by_id": {r["id"]: r["group"] for r in rows}}
    validate_split(rows, manifest)
    Path(split_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(split_path).exists():
        frozen = json.loads(Path(split_path).read_text(encoding="utf-8"))
        if frozen != manifest:
            raise ValueError("Domain preparation differs from the frozen manifest; do not silently resplit.")
    else:
        Path(split_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_domain(split, source=SOURCE, split_path=SPLIT, uci_path=DATA):
    if split not in ("train", "dev"):
        raise ValueError("Domain test split is reserved for final frozen evaluation.")
    manifest = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if (hashlib.sha256(Path(source).read_bytes()).hexdigest() != manifest["source_sha256"]
            or hashlib.sha256(Path(uci_path).read_bytes()).hexdigest() != manifest["protected_uci_sha256"]):
        raise ValueError("Domain source or exclusion corpus differs from the frozen manifest.")
    groups = manifest["group_by_id"]
    rows = [{**r, "group": groups[r["id"]]} for r in source_rows(source) if r["id"] in groups]
    validate_split(rows, manifest)
    selected = set(manifest["partitions"][split])
    return [r for r in rows if r["id"] in selected]
