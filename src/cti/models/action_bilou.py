"""Pure action BIO/BILOU conversion; no tokenizer or ML runtime dependency."""

from cti.models.base import validate_training_data


INTERNAL_TAGS = frozenset(("O", "B-ACTION", "I-ACTION", "L-ACTION", "U-ACTION"))


def bio_to_bilou(tags):
    validate_training_data([[""] * len(tags)], [tags])
    if any(tag not in ("O", "B-ACTION", "I-ACTION") for tag in tags):
        raise ValueError("The BILOU experiment accepts action BIO labels only.")
    output = []
    for i, tag in enumerate(tags):
        continues = i+1 < len(tags) and tags[i+1] == "I-ACTION"
        if tag == "O":
            output.append("O")
        elif tag == "B-ACTION":
            output.append("B-ACTION" if continues else "U-ACTION")
        else:
            output.append("I-ACTION" if continues else "L-ACTION")
    return output


def bilou_to_bio(tags):
    """Normalize unconstrained BILOU while preserving explicit closes.

    B/U begin; I/L continue only an open B/I, otherwise begin. L/U close. An
    unfinished B/I span closes at O/B/U or sentence end. Gold encoding is strict;
    this deterministic normalization applies only to predicted internal paths.
    """
    output, opened = [], False
    for tag in tags:
        if tag not in INTERNAL_TAGS:
            raise ValueError("Unknown internal action BILOU tag.")
        if tag == "O":
            output.append("O")
        else:
            output.append("B-ACTION" if tag[0] in ("B", "U") or not opened else "I-ACTION")
        opened = tag[0] in ("B", "I")
    return output


def bilou_is_valid(tags):
    opened = False
    for tag in tags:
        if tag not in INTERNAL_TAGS or (opened and tag[0] not in ("I", "L")) or (not opened and tag[0] in ("I", "L")):
            return False
        opened = tag[0] in ("B", "I")
    return not opened
