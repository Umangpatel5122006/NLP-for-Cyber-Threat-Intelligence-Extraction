import yaml
from spacy.util import filter_spans

from cti.preprocessing.tokenizer import _get_nlp


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
    doc = _get_nlp(lang)(text)
    spans = []
    for e in entities or []:
        label = label_map[e["label"]]
        if keep is not None and label not in keep:
            continue
        span = doc.char_span(e["start"], e["end"], label=label,
                             alignment_mode="expand")
        if span is not None:
            spans.append(span)
    doc.ents = filter_spans(spans)
    return [
        (t.text, t.idx, t.idx + len(t.text),
         "O" if t.ent_iob_ == "O" else f"{t.ent_iob_}-{t.ent_type_}")
        for t in doc if not t.is_space
    ]