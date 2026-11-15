# Action boundary features and updated default model

Date: 2026-11-15 (Asia/Calcutta)

## Result and app review

The default action model now uses the selected classical boundary-feature CRF.
All-dev exact F1 improves from **0.1379 to 0.1551**. The entity model, tokenizer,
label schema, IOC rules and frontend behavior are unchanged.

**Restart the running review server** to load the new artifact; the server caches
its models. Stop it with Ctrl+C, then from the project root run:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.web'
```

Open http://127.0.0.1:8765, load the example and analyze it. No npm installation
or new dataset download is needed.

This is a modest development improvement. It is not final-test evidence or a
claim of statistically established generalization. Spurious predictions still
need work.

## Classical NLP concepts added

`src/cti/models/action_crf.py` adds an opt-in profile around the existing linear
chain CRF. It does not modify the original `crf.py` feature extractor.

- **Morphology and lexical rules:** a disclosed English suffix normalizer,
  fixed common verb/irregular-form mappings, auxiliary/modal/preposition/
  conjunction/negation cues, neighboring stems and function-word classes,
  class bigrams and bounded clause-boundary distance.
- **Learned boundary dictionary:** stem-level frequency and smoothed rates for
  action starts, membership and ends. Counts come only from each fold's fit
  tokens/tags. Training features subtract the current sentence's counts;
  held-out prediction uses the complete fitted dictionary without updating it.
  A remaining frequency of at least five is required.

These word classes are heuristic cues, not POS predictions. The normalizer is
not a complete English lemmatizer. No trained spaCy NER/POS/parser, neural
embedding, transformer or LLM API is introduced. spaCy remains a blank tokenizer/
shape/rule-matching dependency; all estimation is classical.

Both candidate profiles retain the baseline's existing gazetteer, trigger,
orthographic, affix, IOC-pattern and context features. The previous experiment's
unhelpful dictionary removal/wider context is not carried forward.

## Selection protocol and actual results

`configs/action_features_experiment.yaml` fixes three candidates: baseline,
morphology, and morphology plus the fold-fitted boundary dictionary. All keep
baseline CRF regularization and iteration budget.

Selection uses three report-grouped folds over the **70 official train reports /
8,635 sentences**. Fit uses relevant training sentences only; validation includes
all sentences. Dev/test report IDs are excluded. A candidate needs pooled exact
F1 gain of at least 0.005 and wins in at least two folds.

| Candidate | Fold 1 exact F1 | Fold 2 | Fold 3 | Pooled exact F1 | Pooled relaxed F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.1119 | 0.1401 | 0.1282 | 0.1265 | 0.4446 |
| Morphology | 0.1243 | 0.1346 | 0.1157 | 0.1250 | 0.4591 |
| Morphology + boundary dictionary | 0.1314 | 0.1419 | 0.1231 | **0.1323** | **0.4656** |

The combined profile wins two folds and is selected. That choice is written
before dev rows are loaded. Morphology alone was not scored on dev. The full
baseline and selected model were then fitted on all 3,006 relevant train
sentences and evaluated on all 1,238 dev sentences:

| Metric | Baseline | Selected |
|---|---:|---:|
| Exact precision | 0.1330 | **0.1419** |
| Exact recall | 0.1431 | **0.1711** |
| Exact F1 | 0.1379 | **0.1551** |
| Relaxed F1 | 0.4643 | **0.4966** |
| Token macro/micro F1, excluding O | 0.3957 | **0.4231** |
| Relevant-dev exact F1 (diagnostic only) | 0.1788 | **0.2043** |

There are 104 exact matches vs 87, and 275 missed spans vs 315 under the error
taxonomy. The model also predicts more spans (733 vs 654), including more
spurious spans (400 vs 361). Gains vary by source: APTnotes/MITRE improve most;
Mandiant and Sentinel exact F1 decline slightly.

A conditional paired bootstrap over training reports gives a 95% percentile
interval of **[-0.0031, 0.0128]** for the combined profile's exact-F1 difference.
It includes zero. It holds the OOF fits fixed and omits refitting uncertainty.
Its point difference uses unrounded counts (0.0057), whereas subtraction of
displayed four-decimal CV scores gives 0.0058. No significance claim is made.

Metrics are version 2 on the existing processed BIO labels. This is not a raw
character-span or end-to-end sentence-splitting/defanging benchmark. Dev remains
development material, and the official test split remains reserved.

## Persistence and activation

The profile remains the `crf_tagger` model family and stores its fitted
dictionary inside the joblib artifact. The normal training CLI recognizes
`use_action_morphology` and `use_action_lexicon` for the action layer.
`configs/ner_training.yaml` now enables both for default action training.

Profile artifacts additionally fingerprint `action_crf.py`. Existing baseline
artifacts keep their original dependency set, so the new loader still accepts
them. Tests verify profile round trips, optional source-drift rejection, disabled
flags, empty inputs, action-only labels and normal-CLI training.

The isolated experiment did not replace active files. After independent checks
and positive CV/dev results, a separate activation copied the selected artifact
to `models/ner_action.joblib` and saved its previous binary/manifest at
`models/action_features_v1/previous_active_action.*`. The entity artifact's hash
is unchanged. `promotion.json` records old/new hashes and the restart requirement.

Model files remain ignored by Git. On a fresh checkout, the normal
`python -m cti.train_ner` command regenerates both configured models in a working
runtime. `--layer action` retrains just the updated action model.
Default training reports now use `ner_current_*`, preserving the frozen
`ner_v2_*` baseline files. An explicit `--run-name` remains supported.

## Verification and artifacts

- **155 tests pass in WSL**, including nine new feature/persistence tests:
  `results/action_features_validation.txt`. No new passing native Windows NER
  check is claimed; Windows security settings were not changed.
- Nine CV fits plus full baseline/selected fits completed with exit code 0.
  Saved models retain identical dev predictions after reload.
- The refitted baseline still reproduces every frozen v2 dev prediction.
  Independent rescoring reproduces all nine fold caches and the selection.
- Fresh API inference uses the promoted artifact and valid original evidence.
  Full Chrome UI checks pass against that artifact, including uploads,
  highlighting/keyboard selection, filtering, source evidence, JSON export,
  mobile layout, language fallback and the intentionally induced offline error.
  Browser checks ran on a separate QA port, without stopping the user's server.

`results/action_features_v1/` contains `cv.json`, `selection.json`, `summary.json`,
baseline/selected dev reports and predictions, bootstrap/diagnostic artifacts,
`promotion.json`, compatibility/activation checks, `live_demo.json` and browser
QA evidence. Baseline compatibility was checked before activation; the selected
reference is used for the subsequent active-hash audit.

The previous context/regularization experiment remains historical under
`results/action_cv_v1/`. Source-aware caches invalidate when their dependencies
change; rerunning older experiments under the updated runner may refit them.

The prior experiment commit (`e7af4ae`) has a successful
[GitHub Actions run](https://github.com/Umangpatel5122006/NLP-for-Cyber-Threat-Intelligence-Extraction/actions/runs/43689971420).
The new tests join existing full discovery; the current 155-test changes await
their remote run after commit/push.

## Review and reproduce

```powershell
Get-Content results/action_features_v1/summary.json
Get-Content results/action_features_v1/promotion.json
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -p test_action_features.py -v'
git diff --check
git status --short
```

To reproduce the isolated comparison and its audit:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -u -m cti.experiment_action --config configs/action_features_experiment.yaml --model-dir models/action_features_check --results-dir results/action_features_check'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python scripts/audit_action_experiment.py --results-dir results/action_features_check --model-dir models/action_features_check --config configs/action_features_experiment.yaml --active-action-report results/action_features_v1/action_boundaries_dev.json'
```

Separate check directories preserve this reviewed activation snapshot.
Artifacts contain run metadata/timestamps, so independent retraining can change
binary hashes even when predictions agree. The active-hash audit checks this
local activation snapshot; after an independent retrain, use its matching
training report as the active reference.

Suggested commit: `feat: add classical action boundary features`.
Commits and pushes remain with the user.

Next: reduce background false positives, preferably by selecting relevance/
action gating jointly on training-only out-of-fold predictions. Keep technique
classification and the final frozen test evaluation as separate steps.
