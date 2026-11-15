# NLP for Cyber Threat Intelligence Extraction

An NLP-powered system that extracts cyber threats, attack techniques, vulnerabilities, affected systems, and indicators of compromise (IOCs) from unstructured security reports and threat intelligence feeds.

**Methods constraint:** classical NLP and classical ML only. Active NER models
are CRF, MEMM, and token Logistic Regression. Trained spaCy NER, BiLSTMs,
transformers, neural embeddings, and LLMs are excluded. spaCy's blank tokenizer,
token shapes, and rule-based dictionary matching remain in scope.

**Language support:** English trained NER/relevance/SMS models, plus **English,
Hindi and Hinglish message-pattern rules**. Hindi/Hinglish produce rule Findings
and original-source highlights in both modes. Their trained classifiers and CTI
NER are still unavailable; no multilingual learned accuracy is claimed. The
language control offers Auto/English/Hindi/Hinglish. Automatic script/lexical
guesses are unverified, with manual selection available. IOC extraction works
across surrounding languages. See [the integration checkpoint](checkpoints/HINDI_HINGLISH.md)
and the earlier [coverage checkpoint](checkpoints/LANGUAGE_COVERAGE.md).
The [version 2 phrase-rule update](checkpoints/INDIC_RULES_V2.md) adds verification/
form instructions, bare destinations and bounded account/identity-threat checks,
with warning/negation guards. Trained language support remains unchanged.

---

## Run the Browser Workbench

Frontend and NLP backend start together with one command from the project root.
The launcher checks native Python and the prepared WSL runtime, selecting one
that can load the saved models:

```powershell
python scripts/start_workbench.py
```

Open **http://127.0.0.1:8765**, load the example and select **Analyze report**.
The responsive UI supports original-text highlights, finding filters, source
evidence/components and JSON export. It uses browser HTML/CSS/JavaScript with
the existing Python pipeline; no npm installation is needed. Both modes include
English scam-pattern checks with a visible risk result and source excerpts.
Zero extracted IOCs/entities does not establish that a message is safe.
The evidence map now repeats the risk verdict and highlights its source cues.
The Findings list now combines extraction, message-rule detections and positive
SMS-model predictions through `review_findings`, retaining source evidence.
The trained word/character TF-IDF + linear SVM scores **0.9810 dev spam F1**
on a template-grouped UCI SMS partition (4 missed spam, 0 false positives).
This is not universal scam accuracy: it misses the supplied money/SMS message,
which the rule layer flags. The UI preserves and explains that disagreement.
Rules v2 no longer require a complete phone number: shortening, masking or
removing the target preserves supported financial-offer/request detections.
The current robustness check covers 768 generated combinations and both real
API modes. Restart the server and verify the **English rules v2** label.
Current CTI dev exact-span F1 remains 0.2596 (entities) and 0.1551 (actions).
A [train-only relevance-filter comparison](checkpoints/ACTION_RELEVANCE_EXPERIMENT.md)
retains the current action pipeline: none of four classical cutoffs meets the
fixed gain/recall/coverage policy. No dev or test examples were evaluated in
that comparison; no default artifact changed.
A subsequent [background-training comparison](checkpoints/ACTION_BACKGROUND_EXPERIMENT.md)
also retains the current model: capped/all background reduce false action spans
but lower training-CV exact F1 to 0.1142 / 0.0927 from 0.1323. Neither passes
the recall/gain policy, so dev/test evaluation and activation were skipped.
A [train-only error audit](checkpoints/ACTION_ERROR_AUDIT.md) now separates missed
actions from boundary errors and provides local review CSVs. BIO regeneration
matches the saved training data; 21 repeated-token groups have conflicting action
labels. No automatic label correction or model change was made.
A subsequent [classical ending-feature comparison](checkpoints/ACTION_ENDING_EXPERIMENT.md)
passes training-CV selection but fails dev confirmation (exact F1 0.1551 to
0.1380). That candidate is held; the current model remains active.
See [the current robustness checkpoint](checkpoints/MESSAGE_ROBUSTNESS.md),
[the integrated checkpoint](checkpoints/INTEGRATED_FINDINGS_AND_SMS_MODEL.md)
and [SMS data provenance](docs/SMS_DATASET.md).

A [learned expansion candidate](checkpoints/LEARNED_MESSAGE_EXPANSION.md) now
adds source-audited labeled SMS data. Supplemental dev spam F1 improves from
0.6400 to 0.9538, while UCI dev F1 changes from 0.9810 to 0.9761. The candidate
is **held** because diagnostic false alarms increase; the default stays unchanged.
It still misses the exact reported money/SMS wording without the rule layer.
Preview it independently with
`python scripts/start_workbench.py --port 8766 --message-model models/message_generalization_v1/candidate.joblib`.
See [supplement provenance and data needs](docs/SMS_DOMAIN_DATA.md) before
collecting more examples. Both SMS holdouts and the CTI test remain reserved.
The next [data-intake step](checkpoints/SMS_DATA_INTAKE.md) provides a local
annotation CSV, overlap/review checks and immutable new batch splits. Follow
[the collection guide](docs/SMS_COLLECTION.md) to add real reviewed examples;
`data/local_sms/` stays ignored. Intake checks do not train or change the app.

On a fresh clone, model binaries are absent because `models/` is ignored.
After installing pinned dependencies or preparing WSL, run
`python scripts/start_workbench.py --train-models` to train missing/incompatible
NER and SMS artifacts on train/dev and start the app. The SMS path downloads
the pinned official corpus. `--train-message-model` explicitly prepares just
the SMS model when unavailable. `--check` checks readiness
without starting or training. NER setup failures now have visible instructions
and a retry button; unavailable modes remain selectable for inspection.

For IOC-only use on native Windows:

```powershell
.\.venv\Scripts\python.exe -m cti.web --ioc-only
```

Press Ctrl+C to stop. See [the current fix checkpoint](checkpoints/MESSAGE_ROBUSTNESS.md)
for message-risk behavior, setup recovery and validation; the earlier
[browser checkpoint](checkpoints/BROWSER_WORKBENCH.md) preserves the initial API/UI work.

---

## Setup and Baseline Checks

Generated logs, screenshots, API/demo outputs, per-row predictions and feature
caches stay local through `.gitignore`. Compact experiment reports and the
frozen baseline required by the audit script remain versioned. Required data,
split IDs, code, configuration and checkpoint notes are retained. See
[the output tracking policy](results/README.md).

The core dependency snapshot targets **Windows x64 with Python 3.11**. Run
these commands from the repository root; activating the environment is optional:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pip check

# Configured preprocessing + IOC pipeline
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_pipeline.py -v

# Full suite, including classical NER and evaluation tests
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

`requirements.txt` contains the direct core dependencies;
`requirements-lock.txt` records their resolved dependencies and build tools.
`pyproject.toml` reads the requirements files for package installation.
The configured document pipeline imports its selected built-in modules on demand.
Additional custom modules can register themselves before the pipeline is built.

Historical neural experiments are disabled before their dependencies load.
They are excluded from active comparisons and tests; there is no neural package
extra. Their old dependency list is preserved under `docs/history/`.
Split-lock tests use synthetic data instead of loading held-out test annotations.

**Current validation:** all **310 tests pass** in an isolated Ubuntu-24.04 WSL
environment with Python 3.12.3 and the pinned core dependencies. This includes
saved-NER, original-offset, merger, API, action-feature, scam-pattern and launcher tests;
see `results/action_ending_tests.txt` (local generated log) and
[the working WSL commands](docs/WSL_SETUP.md). Windows Application Control still
blocks native spaCy imports in the Windows environment. No security settings
were changed. A focused native Windows run passes 82 API/language/Indic-rule/
combined-review/intake tests. Actual Chrome checks cover real NER/IOC inference, scam
warnings, JSON export, mobile layout, missing-model recovery and language scope
in both modes. The latest [rule-update checkpoint](checkpoints/INDIC_RULES_V2.md)
records broader requests and the limits of the downloaded development examples.
The earlier [Hindi/Hinglish checkpoint](checkpoints/HINDI_HINGLISH.md)
records phrase-rule support, data auditing and the remaining supervised-model gap.
The earlier [data intake checkpoint](checkpoints/SMS_DATA_INTAKE.md) records
22 native intake tests and private batch leakage checks. The earlier
[learned comparison checkpoint](checkpoints/LEARNED_MESSAGE_EXPANSION.md)
records the candidate, independent audit and release hold. The earlier
[fix checkpoint](checkpoints/MESSAGE_ROBUSTNESS.md)
records phone-independent request matching, mutation checks and remaining limits.
The [boundary-feature checkpoint](checkpoints/ACTION_BOUNDARY_FEATURES.md)
records the selected action profile and active-model upgrade. The earlier
[action checkpoint](checkpoints/ACTION_EXPERIMENT.md) records the negative
context/regularization comparison. The
[browser checkpoint](checkpoints/BROWSER_WORKBENCH.md) records the local UI/API
and its validation. The [merge checkpoint](checkpoints/FINDING_MERGE.md) records consolidated
findings. The [offset checkpoint](checkpoints/OFFSET_MAPPING.md) records original
evidence projection; the [NER checkpoint](checkpoints/NER_INTEGRATION.md) records
real training, dev results and document inference.

**Evaluation correction:** new reports use metrics version 2, which excludes
`O` from token scores and uses one-to-one relaxed-span matching. Token and
relaxed figures in the historical tables below need recalculation. New saved-NER
results below use the corrected metrics and retain their own provenance. See
[the evaluation checkpoint](checkpoints/EVALUATION_FIXES.md) and
[the scoring protocol](docs/paper_protocol.md).

The [Classical NLP tests workflow](.github/workflows/classical-nlp.yml)
checks the complete classical suite on Ubuntu/Python 3.11 after a push or pull
request and can also be run manually in Actions. It checks tokenizer startup,
excludes neural packages, and records test/train-dev validation artifacts.
The committed action experiment passed its classical validation workflow in
[GitHub Actions](https://github.com/Umangpatel5122006/NLP-for-Cyber-Threat-Intelligence-Extraction/actions/runs/43689971420).
The new message-risk/menu tests join normal full-suite discovery; this step awaits its
remote run after commit/push.

---

## Architecture & Pipeline

The pipeline follows a modular, plug-in architecture configured via `configs/config.yaml`:
1. **Cleaner (`cleaner`)**: Unicode normalization (NFC), strips non-essential invisible characters (`\u200b`, `\ufeff`) while preserving Indic characters for cross-lingual support.
2. **Defanger (`defanger`)**: Normalizes obfuscated indicators (e.g., `hxxps://`, `evil[.]com`, `[:]`, `[at]`) back to canonical form while preserving the original raw text in `doc.metadata["text_before_defang"]`.
3. **Language Detector (`language_detector`)**: Identifies document script and language (`en`, `hi`, `mr`, `gu`).
4. **Sentence Splitter (`sentence_splitter`)**: Sentence boundary detection aware of IP addresses, domain names, CTI abbreviations (e.g., `e.g.`, `i.e.`), and the Devanagari danda.
5. **Message risk (`message_risk`)**: Separate English rule combinations for money/contact, credential and advance-fee lures, with original evidence. No match is not a safety verdict.
6. **IOC Extractor (`ioc_finder` / `ioc_regex`)**: Rule-based indicator extraction for IPs, domains, URLs, hashes, CVEs, and emails.
7. **Classical NER Taggers**: Models implement the shared `BaseTagger` interface and can be selected by registry name: `crf_tagger`, `memm_tagger`, and `token_logreg`. Each is trained on token-aligned BIO tags across two specialized layers:
   - `entities` layer: `THREAT_ACTOR`, `MALWARE_TOOL`, `INFRASTRUCTURE`
   - `action` layer: `ACTION`
8. **Result Merger (`result_merger`)**: Builds consolidated `findings` and typed
   `overlaps`, retaining raw `entities`, original positions and per-source
   evidence. It runs after all detectors in the configured pipelines.

---

## Saved Sentence Relevance Baseline

The word unigram/bigram TF-IDF + balanced Logistic Regression baseline is now
available as `SentenceRelevanceClassifier` in `src/cti/models/relevance.py`.
Training and prediction both apply the existing cleaner and defanger.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cti.train_relevance
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_relevance.py -v
```

Training uses the 8,635 official train sentences and evaluates the 1,238 dev
sentences. At the fixed threshold 0.5, relevant-class precision is **0.6265**,
recall **0.7419**, and F1 **0.6793**. This is sentence classification using
dataset-provided sentence boundaries, not end-to-end NER. The test split is not
used, and this result is consistent with the earlier notebook baseline.

The command saves `models/relevance.joblib` with a JSON manifest and records
dev metrics/provenance in `results/relevance_v1_dev.json`, with per-sentence
scores in `results/relevance_v1_dev.predictions.csv`. Saved model files are
ignored by Git and can be regenerated using the command above.

`configs/config_relevance.yaml` adds the saved classifier to the document
pipeline. Each sentence gains `relevance_score` and `relevant` fields; report
text, sentence offsets, and all IOCs are retained. Other languages are explicitly
marked unsupported by this English classifier. Scores describe dataset
relevance and are not calibrated probabilities that an attack occurred.

See [the relevance checkpoint](checkpoints/RELEVANCE_MODULE.md) for review
commands, limitations, and next steps. The original `configs/config.yaml`
remains usable without a trained model.

---

## Saved Classical NER and Report Analysis

Entity and action CRFs can now be trained, saved, reloaded and used in the
document pipeline. The example below uses the prepared WSL environment from
PowerShell at the repository root:

```powershell
# Models are already trained locally; regenerate only when needed.
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.train_ner'

# Analyze a report with preprocessing, IOC extraction and both saved NER layers.
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.analyze --input examples/sample_report.txt --output results/ner_current_demo.json'
Get-Content results/ner_current_demo.json
```

In another working pinned environment, run the same `cti.train_ner` and
`cti.analyze` modules with that environment's Python. Retrain there: saved NER
models reject incompatible runtime versions or changed feature resources.

Both CRFs train on 3,006 relevant train sentences and evaluate **all 1,238 dev
sentences**, with no relevance gate. The official test split is unused.

| Layer | All-dev exact P | All-dev exact R | All-dev exact F1 | All-dev relaxed F1 | All-dev token macro F1 |
|---|---:|---:|---:|---:|---:|
| Entities | 0.2470 | 0.2735 | 0.2596 | 0.3258 | 0.2865 |
| Action + boundary features | 0.1419 | 0.1711 | 0.1551 | 0.4966 | 0.4231 |

These metrics version 2 results describe the current saved models. Action
boundaries and spurious spans still need improvement. Evaluation uses the
existing processed BIO labels, rather than raw-span or end-to-end report
annotations. `results/ner_v2_*` preserves the earlier baseline;
`results/action_features_v1/action_boundaries_dev.json` records the selected
action model. Normal retraining writes `ner_current_*` reports to preserve that
history. Saved models reproduce their dev predictions after reload.

`configs/ner_training.yaml` selects training settings; `configs/config_ner.yaml`
selects document modules. Artifacts and manifests are saved under ignored
`models/ner_{entities,action}.*`. The sample output retains both IOC and NER
evidence. NER confidence is `null`. Existing offsets refer to normalized
`document.text`; new `original_start`, `original_end` and `original_text` fields
project every entity and sentence back to the supplied report. The document
also retains the complete `original_text`. Positions count Unicode code points
and use half-open intervals; source evidence uses covering spans when Unicode
composition or replacements prevent a one-to-one mapping. See
[the offset checkpoint](checkpoints/OFFSET_MAPPING.md) for the JSON contract and
[the NER checkpoint](checkpoints/NER_INTEGRATION.md)
for review commands and limitations.

---

## Consolidated Findings and Frontend Readiness

Configured pipelines now keep detector observations in `entities` and provide
a consolidated `findings` view. Exact-span IOC/INFRASTRUCTURE matches carry both
sources as evidence. Actual URL/email host domains are linked as components;
actions and their nested entities remain separate. Conflicting labels and other
overlaps remain available for review. Per-source confidence is preserved without
an aggregate attack probability.

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.analyze --input examples/merge_report.txt --output results/merge_report_demo.json'
Get-Content results/merge_report_demo.json
```

That synthetic report produces 13 observations consolidated into 9 findings,
including 7 primary findings and 2 host components. Repeated occurrences keep
separate positions. Canonical IOC values are additional fields and preserve
the source spelling in evidence. Mixed-case IOC source alignment is also fixed;
URL path/query/fragment case is preserved.

The browser workbench now provides report input, original-text highlighting,
findings and source-evidence review using this schema. Model accuracy
improvements and the remaining extraction tasks continue separately.
See [the merge checkpoint](checkpoints/FINDING_MERGE.md) for the
schema, merge rules, limitations, test evidence and review commands.

---

## Action Boundary Features

The selected action CRF adds English inflection/function-word cues and
stem-level boundary statistics fitted on training data only. Current-sentence
counts are excluded when constructing its training features. Existing useful
feature groups remain enabled; no trained POS parser, neural model or extra
dataset is introduced.

Three-fold report-grouped training CV selected the combined profile (pooled
exact F1 **0.1323** vs **0.1265** baseline), with wins in two folds. The choice
was frozen before dev evaluation. All-dev action exact F1 then improved from
**0.1379 to 0.1551**, with 104 exact matches vs 87. Spurious spans increased
from 361 to 400, so reducing background predictions remains important.

The selected artifact is now the default. Restart the running web server to
load it. The previous model is backed up under ignored `models/action_features_v1/`.
See [the feature checkpoint](checkpoints/ACTION_BOUNDARY_FEATURES.md) for the
protocol, uncertainty, tradeoffs and reproduction commands.

---

## Earlier Train-Only Action CRF Comparison

The fixed `configs/action_experiment.yaml` comparison used three report-grouped
folds across the 70 official train reports. It fit on relevant training
sentences and evaluated every sentence in each validation fold.

| Configuration | Pooled training-CV exact F1 | Decision |
|---|---:|---|
| Current baseline | **0.1265** | Retained |
| Wider context, without entity gazetteer/triggers | 0.1208 | Not selected |
| Same context with stronger sparsity | 0.1209 | Not selected |

Neither variant met the predefined minimum-gain and fold-consistency rule.
Selection was frozen before dev evaluation; losing variants were not scored on
dev. The baseline's full-train refit reproduced all 1,238 saved dev predictions
exactly, including all-dev action exact F1 **0.1379**. That experiment made no
active-model change. The later boundary-feature experiment above updates the
action model.

`python -m cti.experiment_action` reproduces/resumes the comparison in a working
classical runtime. See [the action checkpoint](checkpoints/ACTION_EXPERIMENT.md)
and `results/action_cv_v1/` for the protocol, metrics, diagnostics and review
commands. The official test split remains unused.

---

## Historical Relevant-Only Model Comparison (Dev Set)

`notebooks/12_relevant_only.py` compares NER taggers trained on relevant
sentences only. The first score evaluates all dev sentences, while the
relevant-only score evaluates just the 434 relevant dev sentences (903 gold
entities). This distinction matters: the conditional score does not include
the relevance classifier's false negatives.

| Model | All-dev P | All-dev R | All-dev F1 | Relevant P | Relevant R | Relevant F1 |
|---|---:|---:|---:|---:|---:|---:|
| CRF (`crf_tagger`) | 0.242 | 0.287 | 0.262 | 0.428 | 0.287 | 0.344 |
| MEMM (`memm_tagger`) | 0.216 | 0.309 | 0.254 | 0.408 | 0.309 | 0.352 |
| Token Logistic Regression (`token_logreg`) | 0.140 | 0.308 | 0.193 | 0.283 | 0.308 | 0.295 |

Per-label relevant-dev F1:

| Model | THREAT_ACTOR | MALWARE_TOOL | INFRASTRUCTURE |
|---|---:|---:|---:|
| CRF | 0.485 | 0.289 | 0.312 |
| MEMM | **0.507** | 0.306 | 0.311 |
| Token Logistic Regression | 0.421 | **0.309** | 0.226 |

The TF-IDF relevance classifiers scored positive-class (relevant) F1 of 0.679
for Logistic Regression and 0.685 for LinearSVC. Filtering CRF output with
those predictions yielded end-to-end all-dev F1 of 0.278 and 0.274,
respectively. This is only a modest improvement over the unfiltered CRF's
0.262 all-dev F1: missed relevant sentences limit the two-stage pipeline.
These results use one dev split of 10 reports and are not a stable estimate
of generalization. The test split was not used.

Run the full comparison from the repository root:

```powershell
.\.venv\Scripts\python.exe notebooks\12_relevant_only.py
```

Choose a layer or model subset without editing the script:

```powershell
.\.venv\Scripts\python.exe notebooks\12_relevant_only.py --layer action
.\.venv\Scripts\python.exe notebooks\12_relevant_only.py --models crf_tagger,memm_tagger
```

Earlier neural results are preserved in
[the historical comparison record](docs/history/neural_comparisons.md).
They are excluded from the active project and future model selection.

---

## Action-Layer Model Comparison (Dev Set)

The same relevant-only training comparison on `action` uses 434 relevant dev
sentences with 608 gold action spans. Scores below are exact-span seqeval
metrics. The test split was not used.

| Model | All-dev P | All-dev R | All-dev F1 | Relevant P | Relevant R | Relevant F1 |
|---|---:|---:|---:|---:|---:|---:|
| CRF (`crf_tagger`) | 0.135 | 0.150 | 0.142 | 0.245 | 0.150 | 0.186 |
| MEMM (`memm_tagger`) | 0.118 | 0.197 | 0.148 | 0.229 | 0.197 | **0.212** |
| Token Logistic Regression (`token_logreg`) | 0.031 | 0.122 | 0.049 | 0.066 | 0.122 | 0.085 |

The TF-IDF relevance classifiers are unchanged because they predict sentence
relevance, not entity labels. Applying their predictions to the CRF's action
tags yields all-dev F1 0.152 (Logistic Regression) and 0.157 (LinearSVC),
compared with 0.142 for the unfiltered CRF. The best conditional action F1 is
0.212 from the MEMM, but this is a single 10-report dev set; treat the
difference as preliminary.

---

## What “Layer” Means

A layer is one independent BIO sequence-labeling task. The current `entities`
layer predicts one of `THREAT_ACTOR`, `MALWARE_TOOL`, `INFRASTRUCTURE`, or `O`
at each token. The separate `action` layer predicts `ACTION` or `O`. Separating
actions prevents their long spans from competing with entity labels in the
same BIO sequence.

Splitting `entities` into `threat_actor`, `malware_tool`, and `infrastructure`
would train three taggers instead of one. Each would predict only `B/I` for its
own label plus `O`; their results would then be merged. This can reduce
competition between labels and can represent overlaps between different entity
types, but it multiplies training and inference work and can produce
overlapping/conflicting predictions that need a merge policy. The current BIO
builder also resolves overlaps within a layer, so independent layers would
require updating the layer config and rebuilding the processed BIO files.
Whether F1 improves is an experiment, not a guarantee.

To run any existing comparison:

```powershell
.\.venv\Scripts\python.exe notebooks\12_relevant_only.py --layer entities
.\.venv\Scripts\python.exe notebooks\12_relevant_only.py --layer action
```

---

## NER Model Input & BIO Tagging

- Taggers in `src/cti/models` consume pre-tokenized sentences and aligned BIO tags via `fit(tokens, tags, dev_tokens=None, dev_tags=None)` and predict tag sequences via `predict(tokens)`.
- **Span Sanitization in `src/cti/core/bio.py`**: When converting character offsets to token-level BIO tags (`to_bio`), leading/trailing whitespace and stray sentence/clause boundary punctuation (`.`, `,`, `;`, `:`, quotes) are trimmed before snapping with `alignment_mode="expand"`. This prevents punctuation tokens from being incorrectly labeled as entities and ensures clean token-level boundaries.
- **Gazetteer Grounding**: Backed by MITRE ATT&CK (`data/external/gazetteer.json`), providing alias matching, campaign detection, and infrastructure grounding without penalizing out-of-vocabulary entities.

---

## Earlier CRF Results (All-Sentence Training)

The results in this section are the original CRF comparison trained with all
training sentences. They are not directly comparable to the later relevant-only
training results above. Use the relevant-only tables above when comparing the
classical CRF, MEMM, and token Logistic Regression models on relevant dev sentences.

The CRF feature extractor in `src/cti/models/crf.py` utilizes:
- Rich orthographic, prefix/suffix (1–4 chars from clean word roots), and shape features.
- Guarded high-efficiency IPv6 / IPv4, hash, and CVE pattern recognition.
- Positive-evidence gazetteer signals across actors, malware, and infrastructure.
- Exact boundary indicators (`BOS`, `EOS`) and bidirectional word bigrams (`bigram`, `+1:bigram`).

### Evaluation Results (Dev Set)

#### Entities Layer (`THREAT_ACTOR`, `MALWARE_TOOL`, `INFRASTRUCTURE`)
| Model / Configuration | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| **Baseline CRF** | 0.365 | 0.131 | 0.192 | 903 |
| **Updated CRF (`crf_tagger`)** | **0.370** | **0.137** | **0.200** | 903 |
| - *THREAT_ACTOR* | **0.446** | **0.258** | **0.327** | 194 |
| - *MALWARE_TOOL* | **0.402** | **0.129** | **0.195** | 303 |
| - *INFRASTRUCTURE* | **0.278** | **0.086** | **0.132** | 406 |

#### Action Layer (`ACTION`)
| Model / Configuration | Precision | Recall | F1 Score | Relative Gain |
|---|---|---|---|---|
| **Baseline CRF** | 0.148 | 0.051 | 0.076 | - |
| **Updated CRF (`crf_tagger`)** | **0.224** | **0.086** | **0.124** | **+63.2% F1** |

---

## Systematic CRF Tuning & Evaluation (Phase 5: M0–M3)

To move beyond noisy single-split dev scores and stale default hyperparameters, a four-stage systematic protocol was executed on the `entities` layer across all 80 train + dev reports (70 train, 10 dev; 9,873 sentences):

### 1. Stage 1: Coarse Random Search (30 Draws)
- Random sampling across optimization algorithms (`lbfgs`, `l2sgd`, `ap`, `pa`, `arow`), regularization penalties ($c_1, c_2$), frequency pruning (`min_freq`), and transition matrices (`all_possible_transitions`).
- **Key Finding**: Weaker regularization ($c_1 < 0.1$) leads to severe overfitting on the 93k sparse features ($F_1 \approx 0.19 - 0.22$). Stronger L1 regularization ($c_1 \approx 1.8 - 3.0$) and frequency filtering (`min_freq` $\ge 3$) cuts 60% of noisy attributes, doubles training speed, and yields highest accuracy.

### 2. Stage 2: 5-Fold Report-Grouped Cross-Validation
Top 5 configurations evaluated across 80 documents using `GroupKFold(groups=doc_id)`:

| Rank | Algorithm | $c_1$ | $c_2$ | `min_freq` | `transitions` | 5-Fold CV Exact F1 | 5-Fold CV Relaxed F1 | 5-Fold CV Token F1 |
|---|---|---|---|---|---|---|---|---|
| **1 (Champion)** | **`lbfgs`** | **1.843** | **0.137** | **3** | **True** | **0.219 ± 0.036** | **0.310 ± 0.034** | **0.449 ± 0.025** |
| 2 | `lbfgs` | 2.002 | 1.502 | 3 | True | 0.217 ± 0.031 | 0.301 ± 0.034 | 0.443 ± 0.024 |
| 3 | `l2sgd` | 0.000 | 0.090 | 5 | False | 0.217 ± 0.037 | 0.305 ± 0.041 | 0.449 ± 0.032 |
| 4 | `l2sgd` | 0.000 | 0.019 | 3 | False | 0.216 ± 0.027 | 0.308 ± 0.026 | 0.448 ± 0.032 |
| 5 | `lbfgs` | 2.955 | 0.022 | 5 | True | 0.211 ± 0.031 | 0.299 ± 0.035 | 0.439 ± 0.020 |

### 3. Stage 3: Feature-Group Ablation Study
Tested across 9 conditions using 5-fold report-grouped cross-validation:
- **Curated Gazetteer**: $\Delta = -0.0135 \rightarrow$ **KEEP (Essential)**. Adding leak-free ATT&CK dictionary signals improves Exact F1 by $+1.35\%$.
- **Orthography**: $\Delta = -0.0114 \rightarrow$ **KEEP (Essential)**. Word shape, title-case, upper-case, and digits contribute $+1.14\%$ F1.
- **Subword Affixes**: $\Delta = -0.0102 \rightarrow$ **KEEP (Essential)**. Character prefixes and suffixes (1–4 chars) contribute $+1.02\%$ F1.
- **Context Window**: $w = \pm 2$ is optimal ($\Delta = -0.0070$ for $w=1$; widening to $w=3$ increases training time by 50% with $\Delta = -0.0045$).
- **Bigrams & Triggers**: $\Delta \approx 0.0000 \rightarrow$ Neutral; bigrams are redundant when window features ($w=\pm 2$) are present.
- **IOC Patterns**: $\Delta = +0.0029 \rightarrow$ Neutral; raw regex patterns can be safely omitted from NER since IOC extraction is handled by a dedicated extractor.

### 4. Stage 4: New Feature Exploration
- **Document Consistency**: Adding unconstrained within-report term frequency and precedence tags degraded performance ($\Delta = -0.0064$) due to over-weighting common non-entity tokens (e.g., 'attack', 'system').
- **Unsupervised Word Clusters**: 50 KMeans clusters trained on domain co-occurrence performed on par with baseline ($\Delta = -0.0029$).

---

## Running Evaluation and Tests

```bash
# Run unit test suite
python -m unittest discover -s tests

# Multi-perspective baseline evaluation (exact, relaxed, token, 5-class errors)
python notebooks/07_compare.py

# Relevant-only comparison with diagnostic subsets
python notebooks/12_relevant_only.py --layer entities --models crf_tagger,memm_tagger

# Stage 1: Coarse random search for CRF
python notebooks/14_tune_crf_stage1.py --layer entities --n-draws 30

# Stage 2: 5-fold report-grouped cross-validation across 80 documents
python notebooks/15_tune_crf_stage2_cv.py --layer entities

# Stage 3: Feature-group ablation study
python notebooks/16_crf_ablation.py --layer entities --algorithm lbfgs --c1 1.843 --c2 0.137 --min-freq 3

# Stage 4: New feature exploration
python notebooks/17_crf_new_features.py --layer entities --algorithm lbfgs --c1 1.843 --c2 0.137 --min-freq 3
```
