# Saved classical NER training and document inference

Date: 2026-11-15 (Asia/Calcutta)

## What is ready

`python -m cti.train_ner` trains the entity and action layers, writes versioned
joblib artifacts and manifests, evaluates dev, then reloads each artifact and
requires identical predictions on the full dev split. The default configuration
is `configs/ner_training.yaml`. It uses CRFs for both layers; the persistence
contract also supports the project's MEMM and token Logistic Regression models.
No neural model is trained or loaded.

`configs/config_ner.yaml` connects preprocessing, IOC extraction, and the saved
`ner_entities` / `ner_action` modules. `python -m cti.analyze` accepts inline
text or a UTF-8 report file and emits the resulting document as JSON.
`examples/sample_report.txt` and `results/ner_demo.json` provide a reviewable
input/output pair. The original IOC-only and relevance configurations remain
available.

## Actual train/dev run

Both CRFs trained on **3,006 relevant train sentences** from the official
70-report train split. Evaluation covers **all 1,238 dev sentences** from ten
reports. The 434 relevant dev sentences are a diagnostic subset only.
The official test split was not evaluated. Development loading explicitly
selects train/dev and rejects any overlap with protected test report IDs, even
if the general test-authorization environment variable is enabled.

Metrics version 2 results:

| Layer | All-dev exact P | All-dev exact R | All-dev exact F1 | All-dev relaxed F1 | All-dev token macro F1 | Relevant-dev exact F1 |
|---|---:|---:|---:|---:|---:|---:|
| Entities | 0.2470 | 0.2735 | 0.2596 | 0.3258 | 0.2865 | 0.3374 |
| Action | 0.1330 | 0.1431 | 0.1379 | 0.4643 | 0.3957 | 0.1788 |

Exact default and strict IOB2 scores agree for these valid BIO predictions.
Token metrics exclude `O`; relaxed matching is one-to-one within label.
The large relaxed/exact gap for actions indicates that span boundaries deserve
attention. This step establishes reproducible saved models, not an accuracy
improvement. Entity settings reuse the earlier classical configuration; action
settings are an explicit untuned baseline. No additional search was performed.

Reports are `results/ner_v2_{entities,action}.{json,csv,md}`, with aligned dev
tokens, gold BIO tags, predicted BIO tags, UIDs and report IDs in the adjacent
`.predictions.json` files. They evaluate the existing processed BIO labels,
including earlier annotation sanitization/token alignment. They do not measure
raw character-span accuracy or end-to-end sentence splitting and defanging.
Historical experiment tables are preserved and should not be compared as if
their preprocessing and scoring protocols were identical.

`results/ner_training_validation.txt` records the completed entity run and an
interrupted initial action attempt. The subsequent action-only run completed
successfully; its output is `results/ner_action_training.txt`. Both saved models
passed the full-dev prediction equality check after reload.

## Artifact and inference contracts

- `models/ner_entities.joblib` and `models/ner_action.joblib`, each accompanied
  by `.manifest.json`, are ignored by Git and can be regenerated.
- Manifests record model parameters, layer labels, Python minor version,
  relevant package versions, feature-resource hashes, training/config/data
  hashes, train/dev report IDs, base commit, and binary SHA-256. Feature-resource
  or runtime drift requires retraining. Hashes of text files normalize CRLF/LF
  so checkout line endings alone do not invalidate the model.
- These are trusted local joblib artifacts. Their hash checks detect accidental
  drift; they do not make arbitrary downloaded pickle content safe to load.
- Processed BIO records join to raw metadata by sentence UID, with explicit
  coverage, duplicate, token-offset, label, alignment and BIO-transition checks.
- NER validates offsets and predictions before updating its findings. Rerunning
  a layer replaces its own findings, preserves IOCs and the other layer, and
  avoids duplicate findings from that source. Cross-source overlaps are retained
  as evidence until a separate merge policy is implemented.
- NER is English-only. Other languages receive `unsupported_language` metadata.
- Hard relevance gating is **off by default**. Optional gating requires boolean
  sentence decisions from the relevance module; it must be evaluated separately.
- NER confidence is JSON `null`, with `confidence_kind: unavailable`. No
  calibrated confidence or marginal score is claimed.
- Every NER span refers to **the current normalized `document.text`**. Metadata
  says `offset_space: document_text`. Original-report offset mapping is still
  pending; a frontend must not apply these positions directly to raw input.

## Validation that actually ran

The preceding committed scope step (`433e971`) has a successful
[GitHub Actions run](https://github.com/Umangpatel5122006/NLP-for-Cyber-Threat-Intelligence-Extraction/actions/runs/43631773684):
62 tests and real relevance train/dev persistence passed on Ubuntu/Python 3.11.

For this step, an isolated Python 3.12.3 environment in the existing
Ubuntu-24.04 WSL distribution uses the same pinned core requirements. Dependency
checking passed, spaCy's blank tokenizer initialized with no trained components,
and **all 78 tests passed**, including 16 new NER tests. See
`results/ner_validation.txt`. Actual CRF, MEMM and token Logistic Regression
round trips, two-layer inference, error handling, UID alignment, synthetic
split-lock enforcement and training output are covered.

The real trained models also processed the sample report successfully. All
output spans match substrings of normalized text; NER confidence is null and IOC
evidence is preserved. `results/ner_integrity_validation.txt` records independent
checks of saved dev predictions, metrics, provenance, split IDs and artifacts.

Windows Application Control still blocks the native spaCy dependency stack in
the Windows environment. No Windows security setting was changed. The working
WSL environment and review commands are documented in [WSL_SETUP.md](../docs/WSL_SETUP.md).
These artifacts require that Python 3.12 environment; Python 3.11 users should
retrain in their own matching environment.

The workflow now includes real entity/action training and document inference.
Its updated YAML, shell commands and embedded Python were checked locally.
**This updated workflow has not yet run remotely**; the successful linked run
applies to its committed predecessor.

## Review this step

From the repository root in PowerShell, using the prepared WSL environment:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -p test_ner.py -v'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.analyze --input examples/sample_report.txt --output results/ner_demo.json'
Get-Content results/ner_demo.json
git diff --check
git status --short
```

Training is already complete. To reproduce it deliberately, use the same
environment with `python -m cti.train_ner` (or `--layer entities` / `--layer action`).
Retraining replaces the local model and matching result files.

Suggested commit: `feat: add saved classical NER training and inference`.
Commits and pushes remain with the user.

Next reviewable step: preserve offsets through cleaning/defanging so extracted
evidence can be highlighted in the original report. Then define the IOC/NER
merge policy before adding a frontend. Better action-boundary features and
technique classification remain separate classical NLP tasks.
