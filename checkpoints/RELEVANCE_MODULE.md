# Saved sentence relevance classifier and pipeline module

Date: 2026-11-15 (Asia/Calcutta)

## Scope

The spaCy native-extension blocker still prevents validation of existing NER
taggers. This checkpoint instead promotes the existing sentence relevance
baseline into a component that can be trained, saved, loaded, and run in the
document pipeline using the working core environment. It uses classical NLP;
there are no neural models, LLM calls, or new datasets in this step.

## Implementation

- `src/cti/models/relevance.py`: `SentenceRelevanceClassifier` provides `fit`,
  `predict_scores`, `predict`, `save`, and `load`. It uses word TF-IDF n-grams
  (1, 2), `min_df=2`, sublinear TF, and balanced Logistic Regression with C=1,
  max_iter=1000, seed=1, threshold=0.5. Both training and prediction use the
  existing cleaner and defanger to keep normalization consistent.
- The fitted vectorizer and classifier are saved together. A manifest records
  artifact kind/version, normalization, parameters, provenance, and the artifact
  SHA-256. Loading validates the manifest/hash and fitted binary classifier.
  Joblib loading is intended for trusted local training artifacts.
- The registered `relevance_filter` loads a saved artifact and annotates each
  sentence with `relevance_score` and `relevant`. It retains all report text,
  sentences, offsets, and IOC entities. Model hash and threshold are recorded
  in document metadata. It does not yet gate a downstream NER model.
- English is supported. Other detected languages receive
  `status=unsupported_language` and null sentence scores/decisions. Unsupported
  language is not treated as an irrelevant classification.
- `src/cti/train_relevance.py`: train/dev-only CLI. It rejects report/UID overlap
  and any protected-test IDs in train/dev. It selects train/dev explicitly even
  if `CTI_ALLOW_TEST=1` is present in the calling shell.
- `configs/config_relevance.yaml`: opt-in preprocessing + IOC + relevance
  pipeline. Run from the repository root after training. The original pipeline
  configuration still requires no saved model.

## Actual train/dev run

| Item | Result |
|---|---:|
| Train reports / sentences | 70 / 8,635 |
| Train relevant sentences | 3,006 |
| Dev reports / sentences | 10 / 1,238 |
| Dev relevant sentences | 434 |
| Positive-class precision | 0.626459 |
| Positive-class recall | 0.741935 |
| Positive-class F1 | 0.679325 |
| True negatives / false positives | 612 / 192 |
| False negatives / true positives | 112 / 322 |
| Threshold | 0.5 |
| Reloaded dev predictions | Identical |
| Official test split used | No |

This reproduces the scale of the existing notebook's 0.679 relevance F1; it is
not a model-quality improvement claim. Evaluation uses dataset-provided
sentence boundaries and the relevant positive class, not entity spans or full
report segmentation. No threshold search, cross-validation, or NER rerun was
performed in this checkpoint.

At threshold 0.5, hard gating would skip 112 of the 434 relevant dev sentences.
That motivates recall/threshold experiments before enforcing a downstream
filter. The current module only records decisions and preserves the evidence.
Probability scores are uncalibrated relevance estimates, not attack likelihood.

## Artifacts

- `models/relevance.joblib`: fitted local model (ignored by Git).
- `models/relevance.manifest.json`: local manifest (ignored by Git).
- `results/relevance_v1_dev.json`: configuration, provenance, split IDs, binary
  metrics, and reload verification. The recorded base commit is supplemented by
  hashes of the training, model, cleaning, and defanging source files.
- `results/relevance_v1_dev.predictions.csv`: dev UIDs, report IDs, gold/predicted
  relevance, and scores; usable for later comparisons without rerunning inference.
- `results/relevance_validation.txt`: dependency check and full-suite output.

## Validation and review

All 16 relevance tests passed, covering normalization consistency, fitted model
round trips, threshold overrides, invalid inputs, artifact corruption, unchanged
vocabulary during prediction, split isolation, and pipeline behavior. A fresh
interpreter verifies that inference does not import spaCy or neural libraries.
The pipeline test retains an IOC even in a sentence predicted irrelevant.

Full discovery: **44 tests passed; three modules failed to import** because
Windows Application Control blocks spaCy's `srsly.ujson` extension. The failing
modules remain `test_bio`, `test_models_preprocessing`, and `test_tokenizer`.
The dependency compatibility check passed. No security settings were changed.

```powershell
# Recreate the local model and regenerate the new relevance reports.
.\.venv\Scripts\python.exe -X utf8 -m cti.train_relevance

# Review the focused tests and source changes.
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_relevance.py -v
git diff --check
git diff --stat
```

To inspect a live report using the saved model:

```powershell
@'
import json
from cti.core.pipeline import Pipeline
document = Pipeline.from_config("configs/config_relevance.yaml").run(
    "APT29 used Mimikatz to steal credentials. Read our website privacy policy."
)
print(json.dumps(document.metadata["sentences"], indent=2))
'@ | .\.venv\Scripts\python.exe -X utf8 -
```

## Next steps

Validate the full suite and existing NER taggers in an approved runtime, such as
a CI runner, then integrate saved entity/action taggers. Compare high-recall
relevance thresholds using downstream extraction quality; do not select a hard
gate from relevance F1 alone. Character n-grams, context, and ATT&CK classification
remain separate future experiments. Old token/relaxed NER figures still need
regeneration under metrics version 2. Keep the test split reserved for final use.
