# Action CRF comparison: retain the baseline

Date: 2026-11-15 (Asia/Calcutta)

## Decision

**Keep the current action model.** Neither predefined variant improved the
train-only, report-grouped cross-validation criterion. The working browser app
and both active NER artifacts are unchanged; no restart or retraining is needed
to keep using them.

This is a completed negative experiment, not an accuracy improvement. The
baseline was selected before dev evaluation. Its independent full-train refit
then reproduced all 1,238 frozen dev predictions exactly, with exact F1
**0.1379**, relaxed F1 **0.4643**, and token macro F1 **0.3957**.

## Why this experiment

The frozen action baseline has 608 gold spans and 654 predicted spans on dev:
87 exact matches, 206 boundary errors, 315 missed spans and 361 spurious spans
under the existing error taxonomy. It predicts 289 action spans in irrelevant
dev sentences. The exact/relaxed gap points to boundary difficulty; wider
context and removal of entity-focused dictionary cues were plausible changes
to test.

Dev errors informed this hypothesis. Dev is therefore development material,
not an untouched generalization estimate. The official test split remains
reserved for final frozen evaluation.

## Fixed comparison and selection

`configs/action_experiment.yaml` defines three CRF configurations:

1. **baseline**: the active action settings, context window 2 and all existing
   feature groups.
2. **action_context**: context window 3, with gazetteer and entity trigger
   features disabled; baseline regularization.
3. **action_sparse**: the same action-context profile with stronger L1
   regularization/frequency pruning, reusing the earlier classical values
   `c1=1.8427`, `c2=0.137`, `min_freq=3`.

All use LBFGS, at most 200 iterations, and existing classical feature switches.
No feature-extractor source or model-policy change was needed.

Each candidate uses three report-grouped folds across the **70 official train
reports / 8,635 sentences**. Each fold fits on relevant sentences from its
training reports and predicts **every sentence** in its validation reports.
Each report enters validation exactly once. Official dev/test reports are
excluded from these folds.

The selection criterion was fixed before execution: pooled out-of-fold strict
exact F1 must improve over baseline by at least 0.005, with wins in at least
two of three folds. Otherwise retain baseline. No dev metric enters selection,
and `selection.json` is written before dev rows are selected for evaluation.

| Candidate | Fold 1 exact F1 | Fold 2 | Fold 3 | Pooled exact F1 | Pooled relaxed F1 | Fold wins vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.1119 | 0.1401 | 0.1282 | **0.1265** | 0.4446 | — |
| Action context | 0.1124 | 0.1374 | 0.1128 | 0.1208 | 0.4519 | 1/3 |
| Action sparse | 0.1197 | 0.1328 | 0.1095 | 0.1209 | 0.4467 | 1/3 |

Wider context slightly increased relaxed overlap for one variant but did not
improve exact boundaries. The sparse variant also failed the criterion.
The losing candidates were **not evaluated on dev**.

A paired bootstrap over whole training reports (1,000 resamples, seed 42)
gives conditional 95% percentile intervals for the pooled exact-F1 difference:

| Candidate minus baseline | Difference | Conditional interval |
|---|---:|---:|
| Action context | -0.0057 | [-0.0129, 0.0016] |
| Action sparse | -0.0056 | [-0.0137, 0.0025] |

These intervals describe the fixed out-of-fold predictions. They omit model
refitting uncertainty and do not establish final-test performance or prove a
statistically reliable degradation. The predefined selection rule still
retains baseline.

## Code and artifacts

- `python -m cti.experiment_action` runs the fixed comparison and final
  baseline/winner dev confirmation.
- Fold predictions are cached under ignored `models/action_cv_v1/`. Resume
  requires matching config/data/source/runtime fingerprints; cached predictions
  have checksums and BIO/alignment checks. Final model artifacts remain there,
  separate from the active `models/ner_action.joblib`.
- `results/action_cv_v1/cv.json` records all fold/pooled metrics and report IDs;
  `selection.json` records the frozen decision; `summary.json` records dev
  confirmation and the unchanged-active-model decision.
- `baseline_dev.json` and `baseline_dev.predictions.json` retain the refitted
  baseline's dev results, source provenance and reload equality.
- `baseline_diagnostics.json` includes the error counts and 20 seeded boundary
  examples; `paired_cv_bootstrap.json` records the conditional uncertainty check.
- `run.log` ends with exit code 0. `integrity_validation.txt` records independent
  rescoring of all nine fold caches, split checks, source hashes, frozen-dev
  equality and unchanged active model hashes.

The experiment uses the existing processed BIO labels, not raw character-span
gold or an end-to-end evaluation of sentence splitting/defanging.

## Validation completed

**146 tests pass in WSL**, including eight new synthetic-data tests:
`results/action_experiment_validation.txt`. They cover candidate selection,
minimum gain/fold consistency, deterministic ties, early rejection of unsupported
config fields, train-only CV, freezing selection before dev loading, test-lock
enforcement even when the general test environment flag is enabled, cache resume/
corruption handling, persistence and active-directory protection.

The full real comparison completed: nine CV fits and a full-train baseline fit,
with identical dev predictions after saving/reloading. Independent rescoring
reproduces every stored pooled metric. The refit reproduces the previous frozen
dev document/token predictions exactly. Existing entity/action binaries,
dependency pins, data files and earlier results were not modified.

No Windows security settings changed. New tests and model execution ran in the
working WSL environment; no new passing native-Windows NER check is claimed.
The preceding browser commit (`4f943cf`) has a successful
[GitHub Actions run](https://github.com/Umangpatel5122006/NLP-for-Cyber-Threat-Intelligence-Extraction/actions/runs/43686129829).
The new tests join the existing full-discovery workflow; this uncommitted step
has not yet run remotely.

## Review

```powershell
Get-Content results/action_cv_v1/summary.json
Get-Content results/action_cv_v1/baseline_diagnostics.json
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -p test_action_experiment.py -v'
git diff --check
git status --short
```

To reproduce or resume the experiment deliberately:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -u -m cti.experiment_action'
```

Matching fold caches are reused; final confirmation is refitted. A fresh clone
without ignored caches performs the full comparison. No extra dataset download
is needed for this step.

Then regenerate the independent rescoring, bootstrap and boundary examples:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python scripts/audit_action_experiment.py'
```

Suggested commit: `experiment: compare action CRF context and regularization`.
Commits and pushes remain with the user.

Next: retain the helpful existing feature groups and test action-specific
lexical/morphological boundary features, using train-only selection again.
Review annotation-boundary consistency alongside that work; a larger context
window alone did not solve the problem. Technique classification remains a
separate classical NLP task.
