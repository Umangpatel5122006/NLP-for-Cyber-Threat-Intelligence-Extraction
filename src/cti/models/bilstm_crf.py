from collections import Counter

import torch
import torch.nn as nn
from torchcrf import CRF

from cti.core.registry import register
from cti.models.base import (
    BaseTagger,
    repair_bio_sequence,
    validate_token_sequences,
    validate_training_data,
)


class _Net(nn.Module):
    def __init__(self, vocab_size, n_tags, emb=100, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(2 * hidden, n_tags)
        self.crf = CRF(n_tags, batch_first=True)

    def emissions(self, x):
        h, _ = self.lstm(self.drop(self.emb(x)))
        return self.out(self.drop(h))

    def loss(self, x, y, mask):
        return -self.crf(self.emissions(x), y, mask=mask, reduction="mean")

    def decode(self, x, mask):
        return self.crf.decode(self.emissions(x), mask=mask)


@register("bilstm_crf")
class BiLSTMCRF(BaseTagger):
    name = "bilstm_crf"

    def __init__(self, epochs=10, batch_size=32, lr=1e-3, min_freq=2, seed=1):
        if epochs < 1 or batch_size < 1 or min_freq < 1 or lr <= 0:
            raise ValueError(
                "epochs, batch_size, min_freq, and lr must be positive."
            )
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_freq = min_freq
        torch.manual_seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = None

    def _encode(self, sentences):
        return [[self.w2i.get(w.lower(), 1) for w in s] for s in sentences]

    def _pad(self, seqs, pad=0):
        if not seqs or any(not seq for seq in seqs):
            raise ValueError(
                "Padding requires a non-empty batch of non-empty sequences."
            )
        n = max(len(s) for s in seqs)
        x = torch.tensor([s + [pad] * (n - len(s)) for s in seqs], device=self.device)
        mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs],
                            dtype=torch.bool, device=self.device)
        return x, mask

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
            raise ValueError("Cannot fit the BiLSTM-CRF on an empty training set.")
        tokens, tags = map(list, zip(*nonempty))

        counts = Counter(w.lower() for s in tokens for w in s)
        self.w2i = {"<pad>": 0, "<unk>": 1}
        for w, c in counts.items():
            if c >= self.min_freq:
                self.w2i[w] = len(self.w2i)
        self.tag_list = sorted({t for s in tags for t in s})
        self.t2i = {t: i for i, t in enumerate(self.tag_list)}

        X = self._encode(tokens)
        Y = [[self.t2i[t] for t in s] for s in tags]
        self.net = _Net(len(self.w2i), len(self.tag_list)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.net.train()
            order = torch.randperm(len(X)).tolist()
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                x, mask = self._pad([X[j] for j in idx])
                y, _ = self._pad([Y[j] for j in idx])
                loss = self.net.loss(x, y, mask)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            print(
                f"  BiLSTM-CRF epoch {epoch + 1}/{self.epochs}  "
                f"loss {total / len(X):.3f}"
            )

    def predict(self, tokens):
        validate_token_sequences(tokens)
        if self.net is None:
            raise RuntimeError("BiLSTMCRF must be fitted before calling predict().")
        self.net.eval()
        X = self._encode(tokens)
        out = [[] for _ in tokens]
        nonempty_indices = [i for i, sequence in enumerate(X) if sequence]
        with torch.no_grad():
            for i in range(0, len(nonempty_indices), 64):
                indices = nonempty_indices[i:i + 64]
                x, mask = self._pad([X[index] for index in indices])
                for index, seq in zip(indices, self.net.decode(x, mask)):
                    out[index] = repair_bio_sequence(
                        [self.tag_list[tag_id] for tag_id in seq]
                    )
        return out
