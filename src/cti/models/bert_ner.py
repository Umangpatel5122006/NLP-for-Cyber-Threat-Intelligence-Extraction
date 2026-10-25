import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)

from cti.core.registry import register
from cti.models.base import BaseTagger


@register("bert_ner")
class BertNER(BaseTagger):
    name = "bert_ner"

    def __init__(self, model_name="distilbert-base-cased", epochs=3,
                 batch_size=16, lr=5e-5, max_length=256):
        self.model_name, self.epochs = model_name, epochs
        self.batch_size, self.lr, self.max_length = batch_size, lr, max_length

    def fit(self, tokens, tags, dev_tokens=None, dev_tags=None):
        self.tag_list = sorted({t for s in tags for t in s})
        self.t2i = {t: i for i, t in enumerate(self.tag_list)}
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=len(self.tag_list),
            id2label=dict(enumerate(self.tag_list)), label2id=self.t2i)

        # BERT splits words into sub-pieces; label only the first piece of each word
        enc = self.tok(tokens, is_split_into_words=True, truncation=True,
                       max_length=self.max_length)
        items = []
        for i, words_tags in enumerate(tags):
            labels, prev = [], None
            for w in enc.word_ids(i):
                labels.append(self.t2i[words_tags[w]] if w is not None and w != prev else -100)
                prev = w
            items.append({"input_ids": enc["input_ids"][i],
                          "attention_mask": enc["attention_mask"][i],
                          "labels": labels})

        args = TrainingArguments(output_dir="models/bert_tmp", num_train_epochs=self.epochs,
                                 per_device_train_batch_size=self.batch_size,
                                 learning_rate=self.lr, save_strategy="no", report_to="none")
        Trainer(model=self.model, args=args, train_dataset=items,
                data_collator=DataCollatorForTokenClassification(self.tok)).train()

    def predict(self, tokens):
        self.model.eval()
        device = next(self.model.parameters()).device
        out = []
        for i in range(0, len(tokens), 32):
            batch = tokens[i:i + 32]
            enc = self.tok(batch, is_split_into_words=True, truncation=True,
                           max_length=self.max_length, padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**{k: v.to(device) for k, v in enc.items()}).logits
            preds = logits.argmax(-1).cpu().numpy()
            for b, words in enumerate(batch):
                seq, prev = ["O"] * len(words), None
                for pos, w in enumerate(enc.word_ids(b)):
                    if w is not None and w != prev:
                        seq[w] = self.tag_list[preds[b][pos]]
                    prev = w
                out.append(seq)
        return out