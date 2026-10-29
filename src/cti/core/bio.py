import yaml
from spacy.util import filter_spans

from cti.preprocessing.tokenizer import _get_nlp

_BAD_EDGE_PUNCT = {'.', ',', ';', ':', '"', "'", '\u201c', '\u201d', '`'}


def load_label_map(path="configs/labels.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["dataset_mapping"]

def load_layers(path="configs/labels.yaml"):
    """Return {layer_name: set_of_labels} from the config."""
    with open(path, encoding="utf-8") as f:
        return {name: set(labels) for name, labels in yaml.safe_load(f)["layers"].items()}

def to_bio(text, entities, label_map, lang="en", keep=None):
    """Return a list of (token, start, end, tag) with BIO tags.
    keep: optional set of our label names; other labels are ignored."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    doc = _get_nlp(lang)(text)
    spans = []
    for e in entities or []:
        if not isinstance(e, dict):
            raise TypeError("Each entity must be a mapping.")
        source_label = e.get("label")
        try:
            label = label_map[source_label]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Entity label is not present in the label map: {source_label!r}"
            ) from exc
        if not isinstance(label, str) or not label:
            raise ValueError(f"Mapped entity label must be a non-empty string: {label!r}")
        if keep is not None and label not in keep:
            continue
        start, end = e.get("start"), e.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError(f"Entity has invalid character offsets: {e!r}")

        # Clean stray leading/trailing whitespace and punctuation from raw annotations
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        while end > start and text[end - 1] in _BAD_EDGE_PUNCT:
            end -= 1
        while start < end and text[start] in _BAD_EDGE_PUNCT:
            start += 1
        if start >= end:
            continue

        span = doc.char_span(start, end, label=label,
                             alignment_mode="expand")
        if span is not None:
            spans.append(span)
    doc.ents = filter_spans(spans)
    return [
        (t.text, t.idx, t.idx + len(t.text),
         "O" if t.ent_iob_ == "O" else f"{t.ent_iob_}-{t.ent_type_}")
        for t in doc if not t.is_space
    ]
