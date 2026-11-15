"""Pinned UCI SMS corpus and deterministic template-grouped split access."""

import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "models/sms_source/SMSSpamCollection"
SPLITS = ROOT / "data/splits/sms_spam_v1.json"
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"


def download_sms(destination=DATA):
    destination = Path(destination)
    with urlopen(URL, timeout=40) as response:
        archive = response.read(2_000_001)
    if len(archive) > 2_000_000:
        raise ValueError("Unexpected SMS archive size.")
    with ZipFile(io.BytesIO(archive)) as file:
        data = file.read("SMSSpamCollection")
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError("Official SMS corpus changed; review its provenance before use.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


def template_key(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?:https?://|www\.)\S+", " urltoken ", text)
    text = re.sub(r"\b[^\s@]+@[^\s@]+\b", " emailtoken ", text)
    text = re.sub(r"\d+(?:[.,]\d+)*", " numtoken ", text)
    text = re.sub(r"[^\w]+", " ", text)
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def read_rows(path=DATA):
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        label, separator, text = line.partition("\t")
        if not separator or label not in ("ham", "spam") or not text.strip():
            raise ValueError(f"Invalid SMS corpus row {index + 1}.")
        rows.append({"id": f"sms-{index:05d}", "text": text, "label": label, "group": template_key(text)})
    if not rows:
        raise ValueError("SMS corpus is empty.")
    return rows


def create_split(path=DATA, output=SPLITS):
    from sklearn.model_selection import StratifiedGroupKFold

    rows = read_rows(path)
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    folds = list(splitter.split(rows, [r["label"] for r in rows], [r["group"] for r in rows]))
    test, dev = set(folds[0][1]), set(folds[1][1])
    partition = {"train": [], "dev": [], "test": []}
    for i, row in enumerate(rows):
        partition["test" if i in test else "dev" if i in dev else "train"].append(row["id"])
    manifest = {"schema_version": 1, "dataset_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "protocol": "seven_stratified_template_group_folds_seed42; fold0=test, fold1=dev, rest=train",
                "group_policy": "nfkc_casefold_url_email_numeric_punctuation_v1", "partitions": partition}
    validate_split(rows, manifest)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_split(rows, manifest):
    ids = {r["id"] for r in rows}
    partitions = manifest["partitions"]
    if set(partitions) != {"train", "dev", "test"}:
        raise ValueError("SMS split must contain train/dev/test.")
    assigned = [item for values in partitions.values() for item in values]
    if len(assigned) != len(set(assigned)) or set(assigned) != ids:
        raise ValueError("SMS split IDs overlap or do not cover the corpus.")
    group_owner = {}
    for split, values in partitions.items():
        selected = set(values)
        if not selected:
            raise ValueError("SMS partitions cannot be empty.")
        for row in rows:
            if row["id"] in selected:
                if group_owner.setdefault(row["group"], split) != split:
                    raise ValueError("SMS template group leakage across partitions.")


def load_sms(split, path=DATA, split_path=SPLITS):
    # Test access has no runtime override in this development-only loader.
    if split not in ("train", "dev"):
        raise ValueError("SMS test split is reserved for final frozen evaluation.")
    manifest = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("SMS corpus does not match the frozen split.")
    rows = read_rows(path)
    validate_split(rows, manifest)
    selected = set(manifest["partitions"][split])
    return [row for row in rows if row["id"] in selected]
