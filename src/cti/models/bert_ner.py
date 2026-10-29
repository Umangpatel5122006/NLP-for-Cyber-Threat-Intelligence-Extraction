import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)

from cti.core.registry import register
from cti.models.base import (
    BaseTagger,
    repair_bio_sequence,
    validate_token_sequences,
    validate_training_data,
)


def _align_word_labels(word_ids, word_tags, tag_to_id):
    """Assign labels to first subword pieces and ignore continuation pieces."""
    labels = []
    previous_word = None
    for word_id in word_ids:
        if word_id is None or word_id == previous_word:
            labels.append(-100)
        else:
            if word_id < 0 or word_id >= len(word_tags):
                raise ValueError(
                    f"Tokenizer returned out-of-range word index {word_id}."
                )
            try:
                labels.append(tag_to_id[word_tags[word_id]])
            except KeyError as exc:
                raise ValueError(
                    f"Unknown BIO tag {word_tags[word_id]!r} in training data."
                ) from exc
        previous_word = word_id
    return labels


@register("bert_ner")
class BertNER(BaseTagger):
    name = "bert_ner"

    def __init__(self, model_name="distilbert-base-cased", epochs=3,
                 batch_size=16, lr=5e-5, max_length=256):
        if epochs < 1 or batch_size < 1 or max_length < 3 or lr <= 0:
            raise ValueError(
                "epochs, batch_size, and lr must be positive; "
                "max_length must be at least 3."
            )
        self.model_name, self.epochs = model_name, epochs
        self.batch_size, self.lr, self.max_length = batch_size, lr, max_length
        self.model = None

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
            raise ValueError("Cannot fit BERT NER on an empty training set.")
        tokens, tags = map(list, zip(*nonempty))
        self.tag_list = sorted({t for s in tags for t in s})
        if not self.tag_list:
            raise ValueError("Training data does not contain any BIO tags.")
        self.t2i = {t: i for i, t in enumerate(self.tag_list)}
        self.tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if not self.tok.is_fast:
            raise ValueError("BertNER requires a fast tokenizer for word alignment.")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=len(self.tag_list),
            id2label=dict(enumerate(self.tag_list)), label2id=self.t2i)

        # BERT splits words into sub-pieces; label only the first piece of each word
        enc = self.tok(tokens, is_split_into_words=True, truncation=True,
                       max_length=self.max_length)
        items = []
        for i, words_tags in enumerate(tags):
            word_ids = enc.word_ids(i)
            seen_words = {word_id for word_id in word_ids if word_id is not None}
            if seen_words != set(range(len(words_tags))):
                raise ValueError(
                    f"Training sentence {i} exceeds max_length; increase max_length."
                )
            labels = _align_word_labels(word_ids, words_tags, self.t2i)
            items.append({"input_ids": enc["input_ids"][i],
                          "attention_mask": enc["attention_mask"][i],
                          "labels": labels})

        args = TrainingArguments(output_dir="models/bert_tmp", num_train_epochs=self.epochs,
                                 per_device_train_batch_size=self.batch_size,
                                 learning_rate=self.lr, save_strategy="no", report_to="none")
        Trainer(model=self.model, args=args, train_dataset=items,
                data_collator=DataCollatorForTokenClassification(self.tok)).train()

    def predict(self, tokens):
        validate_token_sequences(tokens)
        if self.model is None:
            raise RuntimeError("BertNER must be fitted before calling predict().")
        self.model.eval()
        device = next(self.model.parameters()).device
        out = [[] for _ in tokens]
        nonempty_indices = [i for i, words in enumerate(tokens) if words]
        for i in range(0, len(nonempty_indices), 32):
            indices = nonempty_indices[i:i + 32]
            batch = [tokens[index] for index in indices]
            enc = self.tok(batch, is_split_into_words=True, truncation=True,
                           max_length=self.max_length, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**{k: v.to(device) for k, v in enc.items()}).logits
            preds = logits.argmax(-1).cpu().numpy()
            for b, words in enumerate(batch):
                seq, prev = ["O"] * len(words), None
                seen_words = set()
                for pos, w in enumerate(enc.word_ids(b)):
                    if w is not None and w != prev:
                        if w >= len(words):
                            raise ValueError(
                                f"Tokenizer returned out-of-range word index {w}."
                            )
                        seq[w] = self.tag_list[preds[b][pos]]
                        seen_words.add(w)
                    prev = w
                if seen_words != set(range(len(words))):
                    raise ValueError(
                        "A sentence exceeds max_length and was truncated; "
                        "increase max_length before prediction."
                    )
                out[indices[b]] = repair_bio_sequence(seq)
        return out
