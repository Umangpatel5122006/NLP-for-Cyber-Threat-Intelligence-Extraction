from abc import ABC, abstractmethod
from collections.abc import Sequence

TokenSequences = Sequence[Sequence[str]]
TagSequences = Sequence[Sequence[str]]


def validate_token_sequences(tokens, name="tokens"):
    """Validate sentence/token containers while allowing empty sentences."""
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
        raise TypeError(f"{name} must be a sequence of token sequences.")

    for sentence_index, sentence in enumerate(tokens):
        if isinstance(sentence, (str, bytes)) or not isinstance(sentence, Sequence):
            raise TypeError(f"{name}[{sentence_index}] must be a token sequence.")
        for token_index, token in enumerate(sentence):
            if not isinstance(token, str):
                raise TypeError(
                    f"{name}[{sentence_index}][{token_index}] must be a string."
                )


def validate_training_data(tokens, tags):
    """Validate token/tag alignment shared by all tagger implementations."""
    validate_token_sequences(tokens)
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise TypeError("tags must be a sequence of tag sequences.")
    if len(tokens) != len(tags):
        raise ValueError("tokens and tags must contain the same number of sentences.")

    for sentence_index, (sentence, sentence_tags) in enumerate(zip(tokens, tags)):
        if isinstance(sentence_tags, (str, bytes)) or not isinstance(
            sentence_tags, Sequence
        ):
            raise TypeError(f"tags[{sentence_index}] must be a tag sequence.")
        if len(sentence) != len(sentence_tags):
            raise ValueError(
                f"Sentence {sentence_index}: token count ({len(sentence)}) "
                f"does not match tag count ({len(sentence_tags)})."
            )
        for tag_index, tag in enumerate(sentence_tags):
            if not isinstance(tag, str):
                raise TypeError(
                    f"tags[{sentence_index}][{tag_index}] must be a string."
                )
            if tag != "O" and not (
                tag.startswith("B-") or tag.startswith("I-")
            ):
                raise ValueError(
                    f"tags[{sentence_index}][{tag_index}] is not a BIO tag: {tag!r}."
                )
            label = tag[2:] if tag != "O" else ""
            if tag != "O" and not label:
                raise ValueError(
                    f"tags[{sentence_index}][{tag_index}] has an empty BIO label."
                )
            if tag.startswith("I-") and (
                tag_index == 0
                or sentence_tags[tag_index - 1]
                not in (f"B-{label}", f"I-{label}")
            ):
                raise ValueError(
                    f"tags[{sentence_index}][{tag_index}] is an orphaned I-tag."
                )


def repair_bio_sequence(tags):
    """Repair orphaned I-tags produced by unconstrained sequence decoders."""
    repaired = []
    previous_label = None
    for tag in tags:
        if tag.startswith("I-"):
            label = tag[2:]
            if previous_label != label:
                tag = f"B-{label}"
        repaired.append(tag)
        previous_label = (
            tag[2:] if tag != "O" and tag.startswith(("B-", "I-")) else None
        )
    return repaired


class BaseTagger(ABC):
    """Contract every NER model must follow."""
    name = "base"

    @abstractmethod
    def fit(
        self,
        tokens: TokenSequences,
        tags: TagSequences,
        dev_tokens: TokenSequences | None = None,
        dev_tags: TagSequences | None = None,
    ) -> None:
        """tokens: list of token lists. tags: list of BIO tag lists."""

    @abstractmethod
    def predict(self, tokens: TokenSequences) -> list[list[str]]:
        """Return a list of BIO tag lists, one per sentence."""
