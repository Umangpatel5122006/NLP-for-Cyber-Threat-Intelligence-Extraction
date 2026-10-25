import itertools
import time

import sklearn_crfsuite
from seqeval.metrics import f1_score, precision_score, recall_score

from cti.models.crf import _sentence_features
from cti.models.data import load_bio

LAYER = "entities"
tx, ty = load_bio(LAYER, "train")
dx, dy = load_bio(LAYER, "dev")

Xtr = [_sentence_features(s) for s in tx]     # computed once
Xdv = [_sentence_features(s) for s in dx]

print(f"{'c1':>6}{'c2':>7}{'P':>8}{'R':>8}{'F1':>8}{'sec':>6}")
best = (0, None)
for c1, c2 in itertools.product([0.05, 0.3, 1.0], [0.05, 0.3, 1.0]):
    t0 = time.time()
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2,
                               max_iterations=100, all_possible_transitions=True)
    crf.fit(Xtr, ty)
    pred = crf.predict(Xdv)
    f1 = f1_score(dy, pred)
    print(f"{c1:>6}{c2:>7}{precision_score(dy, pred):>8.3f}"
          f"{recall_score(dy, pred):>8.3f}{f1:>8.3f}{time.time() - t0:>6.0f}")
    if f1 > best[0]:
        best = (f1, (c1, c2))

print("\nBest dev F1:", round(best[0], 3), "with (c1, c2) =", best[1])