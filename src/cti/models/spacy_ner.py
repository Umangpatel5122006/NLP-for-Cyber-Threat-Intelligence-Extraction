import random

import spacy
from spacy.tokens import Doc
from spacy.training import Example, iob_to_biluo
from spacy.util import minibatch

from cti.core.registry import register
from cti.models.base import (
    BaseTagger,
    repair_bio_sequence,
    validate_token_sequences,
    validate_training_data,
)


@register("spacy_ner")
class SpacyNER(BaseTagger):
    name = "spacy_ner"

    def __init__(self, lang="en", epochs=20, dropout=0.2, seed=1):
        if epochs < 1:
            raise ValueError("epochs must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")
        self.lang, self.epochs, self.dropout, self.seed = lang, epochs, dropout, seed
        self.nlp = None

    def fit(self, tokens, tags, dev_tokens=None, dev_tags=None):
        validate_training_data(tokens, tags)
        if (dev_tokens is None) != (dev_tags is None):
            raise ValueError("dev_tokens and dev_tags must be provided together.")
        if dev_tokens is not None:
            validate_training_data(dev_tokens, dev_tags)

        nonempty = [
            (sentence, sentence_tags)
            for sentence, sentence_tags in zip(tokens, tags)
            if sentence
        ]
        if not nonempty:
            raise ValueError("Cannot fit spaCy NER on an empty training set.")
        tokens, tags = map(list, zip(*nonempty))

        rng = random.Random(self.seed)
        nlp = spacy.blank(self.lang)
        ner = nlp.add_pipe("ner")
        labels = sorted({t[2:] for seq in tags for t in seq if t != "O"})
        if not labels:
            raise ValueError("Training data does not contain any entity labels.")
        for label in labels:
            ner.add_label(label)

        examples = []
        for toks, tg in zip(tokens, tags):
            doc = Doc(nlp.vocab, words=toks)
            examples.append(Example.from_dict(doc, {"entities": iob_to_biluo(tg)}))

        nlp.initialize(lambda: examples)
        for epoch in range(self.epochs):
            rng.shuffle(examples)
            losses = {}
            for batch in minibatch(examples, size=32):
                nlp.update(batch, drop=self.dropout, losses=losses)
            print(
                f"  spaCy epoch {epoch + 1}/{self.epochs}  "
                f"loss {losses.get('ner', 0.0):.1f}"
            )
        self.nlp = nlp

    def predict(self, tokens):
        validate_token_sequences(tokens)
        if self.nlp is None:
            raise RuntimeError("SpacyNER must be fitted before calling predict().")
        out = [[] for _ in tokens]
        for sentence_index, toks in enumerate(tokens):
            if not toks:
                continue
            doc = Doc(self.nlp.vocab, words=toks)
            for _, proc in self.nlp.pipeline:
                doc = proc(doc)
            out[sentence_index] = repair_bio_sequence(
                [
                    f"{token.ent_iob_}-{token.ent_type_}"
                    if token.ent_iob_ in ("B", "I")
                    else "O"
                    for token in doc
                ]
            )
        return out
