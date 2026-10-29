import json
from pathlib import Path


def load_bio(layer, split, lang="en", root="data/processed"):
    with open(Path(root) / f"bio_{lang}_{layer}_{split}.json", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("BIO dataset must contain a list of sentence records.")

    tokens, tags = [], []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("tokens"), list):
            raise ValueError(f"BIO record {record_index} must contain a token list.")
        if not record["tokens"]:
            continue

        sentence_tokens, sentence_tags = [], []
        for token_index, item in enumerate(record["tokens"]):
            if (
                not isinstance(item, (list, tuple))
                or len(item) < 4
                or not isinstance(item[0], str)
                or not isinstance(item[3], str)
            ):
                raise ValueError(
                    f"BIO record {record_index}, token {token_index} "
                    "must contain text and a tag at index 3."
                )
            sentence_tokens.append(item[0])
            sentence_tags.append(item[3])
        tokens.append(sentence_tokens)
        tags.append(sentence_tags)
    return tokens, tags
