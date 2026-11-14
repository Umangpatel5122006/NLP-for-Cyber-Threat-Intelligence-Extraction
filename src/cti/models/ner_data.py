"""Join versioned BIO data to raw train/dev metadata by sentence UID."""

import json
from dataclasses import dataclass
from pathlib import Path

from cti.models.base import validate_training_data
from cti.models.ner_artifact import layer_labels


@dataclass
class NERDataset:
    tokens: list
    tags: list
    uids: list
    doc_ids: list
    sources: list
    relevant: list


def load_aligned_ner(layer, split, raw_rows, processed_dir):
    if split not in ("train", "dev"):
        raise PermissionError("The NER development loader accepts train/dev only.")
    allowed = layer_labels(layer)
    path = Path(processed_dir) / f"bio_en_{layer}_{split}.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("BIO data must contain a list of sentence records.")
    raw = raw_rows.to_dict("records")
    by_uid = {row["uid"]: row for row in raw}
    if len(by_uid) != len(raw):
        raise ValueError("Raw sentence UIDs must be unique within a split.")
    result = NERDataset([], [], [], [], [], [])
    seen = set()
    for record in records:
        if not isinstance(record, dict) or record.get("uid") not in by_uid:
            raise ValueError("BIO record has an unknown sentence UID.")
        uid = record["uid"]
        if uid in seen:
            raise ValueError(f"Duplicate BIO sentence UID: {uid!r}.")
        seen.add(uid)
        row = by_uid[uid]
        text = row["sentence_text"]
        relevance = str(row["relevant"]).strip().casefold()
        if not isinstance(text, str) or relevance not in ("yes", "no"):
            raise ValueError("Raw sentences need string text and yes/no relevance labels.")
        if not isinstance(record.get("tokens"), list):
            raise ValueError("BIO sentence must contain a token list.")
        words, tags = [], []
        previous_end = 0
        for item in record["tokens"]:
            if not isinstance(item, list) or len(item) != 4:
                raise ValueError("BIO tokens must contain text/start/end/tag.")
            word, start, end, tag = item
            if (
                not isinstance(word, str) or type(start) is not int or type(end) is not int
                or not previous_end <= start < end <= len(text) or text[start:end] != word
                or not isinstance(tag, str)
            ):
                raise ValueError(f"Invalid BIO token offsets or tag in sentence {uid!r}.")
            if tag != "O" and (tag[:2] not in ("B-", "I-") or tag[2:] not in allowed):
                raise ValueError(f"BIO tag {tag!r} is outside layer {layer!r}.")
            words.append(word)
            tags.append(tag)
            previous_end = end
        result.tokens.append(words)
        result.tags.append(tags)
        result.uids.append(uid)
        result.doc_ids.append(row["doc_id"])
        result.sources.append(row.get("source", "unknown"))
        result.relevant.append(relevance == "yes")
    if seen != set(by_uid):
        raise ValueError("BIO records do not cover every raw sentence UID in the split.")
    validate_training_data(result.tokens, result.tags)
    return result
