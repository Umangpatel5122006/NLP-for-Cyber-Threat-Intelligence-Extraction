# Classical-only methods and CI preparation

Date: 2026-11-15 (Asia/Calcutta)

## Confirmed scope

The user clarified that classical ML is allowed and deep learning is excluded.
The active project uses classical NLP/ML: rules, dictionaries, regular
expressions, TF-IDF, Logistic Regression, linear SVM, CRF, and MEMM.

spaCy's blank tokenizer, token shapes, and rule-based PhraseMatcher remain
allowed. The trainable spaCy NER component initializes a neural model and is
excluded, alongside BiLSTM-CRF, BERT/transformers, neural embeddings, neural
dependency parsers, and LLMs. See the
[spaCy EntityRecognizer API](https://spacy.io/api/entityrecognizer) for its
trainable-model configuration. A dependency library's inclusion does not mean
the project trains or runs all of its available model types.

`AGENTS.md` records this scope for future project work. Older neural-upgrade
suggestions in historical learning/checkpoint notes are superseded.

## Changes

- Centralized the three active NER choices in `src/cti/core/model_policy.py`:
  `crf_tagger`, `memm_tagger`, and `token_logreg`.
- Registry creation/registration rejects the three known neural model names.
  Historical experiment modules are guarded before their dependencies import.
  Bulk module discovery skips them.
- Both comparison scripts use the shared classical model map. The relevant-only
  comparison defaults to the three allowed models, and its obsolete `--epochs`
  option is removed. Neural choices fail before dataset loading or training.
- Removed neural-training tests from the active suite. Five scope tests verify
  registry guards, early import rejection, and CLI choices/defaults.
- Removed the package's neural extra and refreshed installed core metadata.
  Only the eight pinned core dependencies are declared, and PyTorch,
  Transformers, torchcrf, and Accelerate are absent from the core environment.
- Preserved earlier code/results as history. Neural comparison rows moved to
  `docs/history/neural_comparisons.md`; the former optional requirements list
  moved to `docs/history/excluded_neural_requirements.txt`.
- Updated the active roadmap to remove neural upgrades, neural translation,
  embedding encoders, trained spaCy NER, and neural parsing as future work.

## Prepared CI workflow

`.github/workflows/classical-nlp.yml` runs on pushes, pull requests, or manual
dispatch. It follows the standard
[GitHub Python build/test workflow](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).
It creates an isolated Ubuntu/Python 3.11 environment and installs the pinned
core dependency snapshot. The Linux job will validate that package set on Linux;
the earlier successful local installation alone does not establish that result.

The workflow checks that `spacy.blank("en")` has no trained pipeline components
and that neural runtime packages are absent. It then runs the full classical
test suite and the train/dev relevance persistence command. The test split stays
locked; split-lock tests use synthetic authorization fixtures.

Environment details, unittest output, and the CI relevance report/predictions
are uploaded as `classical-nlp-validation`. No model is deployed or published.
The workflow provides an independent runtime for checks currently blocked by
Windows Application Control without changing local security settings.

**Status: prepared, not remotely executed.** YAML structure, four bash blocks,
and embedded Python syntax were validated locally. A passing CI run is still
required before claiming that the complete classical suite passes.

## Local validation

- All five scope-policy tests passed.
- Full discovery: **49 tests passed; three modules failed to import** because
  Windows Application Control blocks spaCy's `srsly.msgpack._packer` extension.
  The affected modules are `test_bio`, `test_models_preprocessing`, and
  `test_tokenizer`.
- Dependency compatibility passed. Installed metadata has no neural extra.
- No new model training, new scores, dataset changes, or Windows security
  changes were made in this checkpoint. Existing relevance artifacts/results
  and earlier NER results are preserved.
- Raw local output is in `results/classical_scope_validation.txt`.

## Review and next action

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_model_policy.py -v
git diff --check
git diff --stat
```

After the user's normal commit/push, open GitHub Actions and inspect **Classical
NLP tests**. Its full-suite log must be assessed before moving past the runtime
blocker. No commit, push, or remote workflow dispatch was performed by the agent.
After that validation, continue saved classical entity/action tagger integration,
development-only diagnostics, and TF-IDF/linear ATT&CK classification.
