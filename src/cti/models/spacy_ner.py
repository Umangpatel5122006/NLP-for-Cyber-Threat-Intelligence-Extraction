import random

import spacy
from spacy.tokens import Doc
from spacy.training import Example, iob_to_biluo
from spacy.util import minibatch

from cti.core.registry import register
from cti.models.base import BaseTagger


@register("spacy_ner")
class SpacyNER(BaseTagger):
    name = "spacy_ner"

    def __init__(self, lang="en", epochs=20, dropout=0.2, seed=1):
        self.lang, self.epochs, self.dropout, self.seed = lang, epochs, dropout, seed

    def fit(self, tokens, tags, dev_tokens=None, dev_tags=None):
        random.seed(self.seed)
        nlp = spacy.blank(self.lang)
        ner = nlp.add_pipe("ner")
        for label in {t[2:] for seq in tags for t in seq if t != "O"}:
            ner.add_label(label)

        examples = []
        for toks, tg in zip(tokens, tags):
            doc = Doc(nlp.vocab, words=toks)
            examples.append(Example.from_dict(doc, {"entities": iob_to_biluo(tg)}))

        nlp.initialize(lambda: examples)
        for epoch in range(self.epochs):
            random.shuffle(examples)
            losses = {}
            for batch in minibatch(examples, size=32):
                nlp.update(batch, drop=self.dropout, losses=losses)
            print(f"  spaCy epoch {epoch + 1}/{self.epochs}  loss {losses['ner']:.1f}")
        self.nlp = nlp

    def predict(self, tokens):
        out = []
        for toks in tokens:
            doc = Doc(self.nlp.vocab, words=toks)
            for _, proc in self.nlp.pipeline:
                doc = proc(doc)
            out.append([f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ in ("B", "I") else "O"
                        for t in doc])
        return out