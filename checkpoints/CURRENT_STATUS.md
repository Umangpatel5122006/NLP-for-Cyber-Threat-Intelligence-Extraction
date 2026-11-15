# Project progress, in order

Latest local checkpoint: [train-only action error audit](ACTION_ERROR_AUDIT.md).
Frozen OOF predictions show 559 exact actions, 1,702 partial overlaps and 2,192
gold actions with no overlap. Boundary pairs more often contain extra tokens;
correct endings are less frequent than correct starts. All training BIO regenerates
exactly, but 21 identical-token groups have conflicting action tags. Local CSVs
provide all 43 conflict records with context and 48 deterministic error examples.
All 299 WSL tests pass; inputs/model hashes are verified. No fitting, label/split
changes or dev/test evaluation occurred. Review conflicts and predeclare a
classical boundary-feature comparison next; neither is a completed model upgrade.

Previous local checkpoint: [action background training](ACTION_BACKGROUND_EXPERIMENT.md).
All six capped/all-background candidate fold fits completed. Both policies reduce
background action predictions but lose too many correct actions: training-CV
exact F1 falls from 0.1323 to 0.1142 / 0.0927. Neither wins any fold or passes
the fixed gain/recall policy. The current model remains active; dev/test were
not evaluated and no new candidate binary was created. All 288 WSL tests and
the independent prediction/sampling audit pass. Active model/config hashes are
unchanged. The subsequent error audit is recorded above.

Previous local checkpoint: [action relevance comparison](ACTION_RELEVANCE_EXPERIMENT.md).
Four predeclared TF-IDF/Logistic Regression cutoffs were compared on three
report-disjoint training folds. None meets the fixed gain/recall/coverage policy;
the baseline stays active. Dev/test were not evaluated in this experiment.
All 279 WSL tests and the independent saved-prediction audit pass. Existing
model hashes and app behavior are unchanged. The background-training comparison
that followed is recorded above.

Previous local checkpoint: [Hindi/Hinglish rules v2](INDIC_RULES_V2.md).
Verification/form requests and bare destinations now support bounded identity/
account-threat findings, preserving warning/negation context and source offsets.
Downloaded authored examples are development diagnostics, not accuracy evidence.
All 270 WSL tests and 82 native focused tests pass. English model weights and the
held candidate are unchanged; no Hindi/Hinglish classifier was trained.

Previous local checkpoint: [Hindi/Hinglish message checks](HINDI_HINGLISH.md).
Both modes apply classical Hindi/Hinglish phrase/request/context checks, show
original-source cues and support Auto/manual language selection. English SMS/CTI
models remain English-only; no new-language learned accuracy is claimed. Intake
accepts en/hi/hi-en with per-language class/group locks. A pinned 184-row source
was audited but not fitted: no Devanagari rows or author language labels. Real
reviewed Hindi/Hinglish data are still needed. All 266 WSL tests and 78 native
focused tests pass. Model weights are unchanged.

Previous local checkpoint: [language and analysis coverage](LANGUAGE_COVERAGE.md).
One NLP target language is supported: English. Script guesses do not add trained
Hindi/Marathi/Gujarati support. API/UI now disclose assumed English, unsupported
language and incomplete checks while retaining IOC/positive-rule evidence.
Non-Latin scripts in a message are no longer ignored in favor of an ASCII URL.
All 253 WSL tests pass; native subsets pass 10 language, 10 review and 23 API
tests. Default and held-candidate browser checks cover both modes. Model weights
are unchanged; no multilingual accuracy gain or final-test evaluation is claimed.

Previous local checkpoint: [reviewed SMS data intake](SMS_DATA_INTAKE.md).
The empty local annotation CSV and audit/freeze command are ready. They exclude
unresolved/synthetic records and prior corpus/diagnostic/campaign overlaps,
preserve reviewer/source provenance locally, freeze new group splits, and reject
test access. No new real messages or training are claimed. Next input is the
team's independently reviewed legitimate and suspicious SMS collection; see
[the guide](../docs/SMS_COLLECTION.md). Private batches stay ignored by Git.
Current validation is 239 WSL tests and 22 native Windows intake tests.

Previous local checkpoint: [learned SMS data expansion](LEARNED_MESSAGE_EXPANSION.md).
The candidate learns from 255 additional overlap-audited training messages and
raises supplement dev spam F1 from 0.6400 to 0.9538 (UCI 0.9810 to 0.9761).
It is **held**, with default artifacts unchanged: authored diagnostic false
alarms rise from 2 to 6 and the exact reported wording still predicts ham.
Rules v2 continue flagging the reported variants. A separate preview is available;
217 WSL tests and both default/preview browser functional checks pass. Preview
QA records its false alarm rather than counting it as correct classification.
No held-out test was evaluated and no new remote CI run is claimed. Broader
independently labeled scam and legitimate financial/delivery data are needed
before another comparison; see [data needs](../docs/SMS_DOMAIN_DATA.md).

Previous checkpoint: [phone-independent message robustness](MESSAGE_ROBUSTNESS.md).
The [unified Findings and trained classical SMS model](INTEGRATED_FINDINGS_AND_SMS_MODEL.md) step is committed as `8a03fa8`.
The earlier visibility work is recorded in [RISK_VISIBILITY_AND_MODEL_SCOPE.md](RISK_VISIBILITY_AND_MODEL_SCOPE.md).
The [scam-message patterns and NER-mode recovery](MESSAGE_RISK_AND_MODE_RECOVERY.md) step is committed.
The [action boundary features and model upgrade](ACTION_BOUNDARY_FEATURES.md) step is committed.
The earlier [train-only action comparison](ACTION_EXPERIMENT.md) step is committed.
The [local browser workbench and API](BROWSER_WORKBENCH.md) step is committed.
The [consolidated findings](FINDING_MERGE.md) step is committed.
The [original-report evidence offsets](OFFSET_MAPPING.md) step is committed.
The [saved classical NER and document inference](NER_INTEGRATION.md) step is committed.
The preceding [classical-only methods and CI](CLASSICAL_SCOPE.md) step is committed.
The [saved sentence relevance module](RELEVANCE_MODULE.md) is committed.
The earlier [evaluation correctness fixes](EVALUATION_FIXES.md) are committed.
New reports use metrics version 2. Historical token and relaxed-span figures
below need recalculation; the original experiment records are preserved.

The relevance baseline now trains on the official train split, saves/reloads
its fitted TF-IDF + Logistic Regression pipeline, and annotates live sentences.
Dev sentence F1 is 0.6793 at threshold 0.5. Entity and action CRFs now also have
saved artifacts, full-dev reload checks and live document modules. New metrics
version 2 all-dev exact F1 is 0.2596 (entities) and 0.1551 (actions), using the
existing processed BIO labels. The official test split is still reserved.
All 270 tests pass in the isolated Ubuntu WSL Python 3.12 environment; Windows
native imports remain blocked by Application Control without policy changes.
Deep learning is excluded. Allowed NER choices are CRF, MEMM, and token Logistic
Regression; historical neural experiments are guarded against execution.
The committed action comparison passed its classical validation workflow
in GitHub Actions. The current robustness changes await their remote run after
commit/push. See the latest checkpoint for evidence and
[WSL_SETUP.md](../docs/WSL_SETUP.md) for working local commands.

Original-report mapping is implemented for cleaning/defanging. The document
preserves `original_text`, and entities/sentences include original evidence
spans alongside normalized offsets. The configured pipelines now add consolidated
`findings` and typed `overlaps`, retaining detector observations and copied source
evidence. All 25 new merger tests also pass on Windows. Mixed-case IOC source
alignment is fixed without changing URL path case.

The local browser workbench now provides report input/file upload, original-text
highlighting, finding filters, source/component details and JSON export. HTML,
CSS and JavaScript are served with a Python API using the existing pipeline.
One command, `python scripts/start_workbench.py`, selects a compatible runtime
and starts both. All 22 API tests also pass on
Windows; actual Chrome checks cover real NER/IOC inference, Unicode uploads,
mobile layout and error states. See the latest checkpoint for the WSL command.

The reported money/SMS lure now shows a prominent suspicious-pattern warning
in both modes, independently of technical IOC/NER findings. English rules
retain original cue evidence and distinguish selected warning/negation contexts.
NER options remain selectable with setup guidance; refresh retries failed loads.
The launcher checks models/runtime, with explicit train/dev training for fresh
clones. Native Windows now also passes 17 rule and 7 combined-review tests. Chrome
checks verify the message, export, mobile display and missing-model UI recovery.
The rule layer remains a disclosed rule baseline rather than fact checking.
The screenshot's plain-newline message reproduces as suspicious with seven cues
and zero technical findings. The evidence map now shows the verdict and clickable
cue highlights directly. The Findings list now includes a message-rule row with
seven evidence cues through the new combined `review_findings` backend contract.
A separately trained word/character TF-IDF + linear SVM is active for short
English SMS. Training-only grouped CV selected it before dev evaluation; dev
spam F1 is 0.9810 (103 TP, 0 FP, 4 FN, 689 TN). Its 797-row holdout was not evaluated.
It misses the supplied money/SMS example; rules retain the suspicious finding
and the UI explicitly shows disagreement. This is not universal scam accuracy.
Twenty-two Chrome groups pass, including shortened, masked and absent targets
in both modes. Rules v2 remove phone validation as a detection prerequisite,
use directive/offer cues and preserve contextual/negation handling. The missing
target still produces one review finding with six cues. The 768 generated
regression combinations pass; SMS dev review-alert metrics remain unchanged.
The UI identifies rules v2 and rejects a stale v1 backend with restart guidance.
Existing CTI NER models/metrics remain unchanged and limited.

The first targeted action comparison is complete: three-fold CV over official
train reports retained the baseline (pooled exact F1 0.1265 vs 0.1208/0.1209).
The decision was frozen before dev loading. A full-train baseline refit then
reproduced all frozen dev predictions exactly; neither active model changed.

The follow-up adds optional action morphology and a training-only boundary
dictionary. The combined profile passed train-only CV (0.1323 vs 0.1265),
then improved dev exact F1 to 0.1551. It is now the active action model, with
the previous artifact backed up. Restart the review server to load it.
Entity inference and the frontend schema remain compatible. The full Chrome
check passed using the promoted artifact.

Next: expand legitimate/scam data coverage beyond the small source-audited
supplement; address false alarms and independently evaluate the SMS/hybrid
assessment. Continue CTI background false-positive reduction
as a separate train-only relevance/action-gating experiment.
More spurious spans remain a tradeoff of the new
model; final-test performance is not yet known.
Model accuracy, technique classification, remaining extraction tasks and final
frozen evaluation still need work. NER confidence remains unavailable (`null`)
and no aggregate confidence is invented during merging.

## 1. Project goal and structure

The project aims to extract threat-related entities, actions, and indicators of compromise (IOCs) from cybersecurity reports. Its core design is modular: components register under names, and a pipeline builds them from configuration so modules can be changed without rewriting the pipeline.

- [pyproject.toml](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/pyproject.toml) sets up the installable Python package.
- [base.py: `Entity`, `Document`, `BaseModule`](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/core/base.py) defines the common document and pipeline-module contract.
- [registry.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/core/registry.py) registers components by name and creates them.
- [pipeline.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/core/pipeline.py) loads registered modules and runs the configured pipeline.

The [basic pipeline](../configs/config.yaml) runs cleaning, defanging, language
guessing, sentence splitting, English message rules, IOC extraction and merging.
The [full workbench pipeline](../configs/config_ner.yaml) also runs the saved
English SMS classifier and entity/action CRFs. IOC workbench mode includes the
optional SMS classifier while skipping those CRFs. Relevance annotation remains
opt-in; hard sentence gating is disabled by default.

## 2. Dataset and label decisions

The project uses the SMAH-CTI dataset: **12,304 sentences from 100 reports**, with 4,302 sentences marked relevant. The reports are divided by report—not by sentence—into 70 train reports, 10 dev reports, and 20 test reports.

- Raw dataset: [SMAH-CTI.json](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/data/raw/SMAH-CTI.json)
- Split IDs and validation files: [data/splits/](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/data/splits)
- Dataset loader: [dataset.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/core/dataset.py)
- Label mappings and layers: [labels.yaml](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/configs/labels.yaml)

The dataset’s labels are mapped as follows:

- `ThreatActor` → `THREAT_ACTOR`
- `MalwareTool` → `MALWARE_TOOL`
- `Action` → `ACTION`
- `Infrastructure_Indicator` → `INFRASTRUCTURE`

`Infrastructure_Indicator` combines several kinds of mentions—software, files, systems, indicators, and other infrastructure-related phrases—so it remains one broad label for now. IOCs such as IP addresses and hashes are handled separately by the IOC extractors.

The dataset notes identify its license as CC BY-NC 4.0, so its use is non-commercial. Details are in the phase notes, especially [PHASE1_README.md](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/checkpoints/PHASE1_README.md).

## 3. Text preprocessing

The configured preprocessing modules are implemented as swappable pipeline components:

1. **Cleaner** performs NFC Unicode normalization, removes selected invisible characters, and collapses whitespace. It keeps Indic-script joiners that can affect written words.
2. **Defanger** converts common defanged text such as `hxxps://`, `[.]`, and `[at]` to normal URL characters.
3. **Language detector** estimates language from script. Latin text is treated as English; Gujarati is identified by its script; Hindi and Marathi are distinguished using a small list of Marathi words.
4. **Sentence splitter** uses rules for punctuation, abbreviations, and the Devanagari danda. It stores sentence text and offsets in document metadata.

Relevant modules are in [src/cti/preprocessing/](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/preprocessing). Their tests cover basic cleaning, defanging, language, sentence-splitting, and tokenization behavior.

**Boundary detail:** defanging changes the text. Pipeline `start`/`end` positions
refer to normalized text; `original_start`/`original_end` now project evidence
through cleaner/defanger changes to the supplied report. See the latest offset
checkpoint for covering-span and Unicode units. The dataset's BIO tags are
still built from the original annotated sentences, not from text run through
the document pipeline.

Language identification is currently a heuristic, not a full language-identification model. English is the training language today; multilingual support will need further validation on Hindi, Marathi, and Gujarati data.

## 4. IOC extraction

Two IOC extractors are available:

- [ioc_regex.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/extractors/ioc_regex.py) is the simpler hand-written regex baseline for IPv4, CVEs, and common hash lengths.
- [ioc_finder.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/extractors/ioc_finder.py) wraps the `ioc-finder` library and handles more types, including IPv6, domains, URLs, emails, CVEs, and hashes.

The configured pipeline uses `ioc_finder`; the regex implementation remains available as an alternative. The phase notes report that IOC extraction matches only a small portion of `Infrastructure_Indicator` annotations, because that label includes many non-IOC concepts as well. See [PHASE3_4_README.md](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/checkpoints/PHASE3_4_README.md).

## 5. Converting annotated spans to BIO training data

NER models need token-level labels, so annotated character spans are converted to `B-LABEL`, `I-LABEL`, and `O` tags.

- [tokenizer.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/preprocessing/tokenizer.py) uses a blank spaCy tokenizer, without requiring a downloaded linguistic model.
- [bio.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/core/bio.py) maps dataset labels, snaps character spans to tokens, cleans stray edge punctuation, and resolves conflicting overlaps with spaCy’s `filter_spans`.
- [05_build_bio.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/05_build_bio.py) builds the processed data.
- The resulting datasets are in [data/processed/](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/data/processed).

The project separates BIO data into two current layers:

- **`entities`** contains `THREAT_ACTOR`, `MALWARE_TOOL`, and `INFRASTRUCTURE`.
- **`action`** contains `ACTION`.

This separation was introduced because long action phrases can overlap shorter entity mentions. The phase notes report that the separated data retains about 99–100% of spans, whereas earlier single-layer experiments lost more infrastructure and malware/tool spans. That reported retention is documented in [PHASE3_4_README.md](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/checkpoints/PHASE3_4_README.md).

## 6. Model interface and model options

The models share `fit(tokens, tags, dev_tokens, dev_tags)` and `predict(tokens)` through [BaseTagger](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/models/base.py). They predict one BIO tag sequence per input sentence.

Current options include:

- **CRF**: [crf.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/models/crf.py) uses `sklearn-crfsuite`, contextual word features, spelling and shape features, cyber-specific patterns, and a MITRE ATT&CK gazetteer.
- **MEMM**: [memm.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/models/memm.py) uses scikit-learn Logistic Regression with previous predicted BIO-tag context.
- **Token Logistic Regression**: in the same file, predicts each token independently, without using previous predicted tags.
- **spaCy NER**: [spacy_ner.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/models/spacy_ner.py) trains spaCy’s NER component from a blank pipeline.
- **BiLSTM-CRF**: [bilstm_crf.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/models/bilstm_crf.py) uses PyTorch embeddings, a bidirectional LSTM, and a CRF output layer.
- **DistilBERT NER**: [bert_ner.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/models/bert_ner.py) is an optional transformer-based alternative. The earlier reported CPU training run took about 2.5 hours, so it was excluded from the more recent multi-model comparison.

Supporting model-data loading and validation are in [data.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/models/data.py). Some current input checks and regression coverage are in [test_models_preprocessing.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/tests/test_models_preprocessing.py).

## 7. Earlier CRF exploration

The notebooks record the model-development process:

- [01_explore.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/01_explore.py), [02_counts.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/02_counts.py), and [03_examples.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/03_examples.py) inspect the dataset and label examples.
- [04_ioc_check.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/04_ioc_check.py) checks IOC extraction against the dataset.
- [06_dropped.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/06_dropped.py) documents span losses from earlier layer experiments.
- [07_compare.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/07_compare.py) compares configured model implementations.
- [08_diagnose.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/08_diagnose.py) examines errors and train-vs-dev behavior for the CRF.
- [09_tune_crf.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/09_tune_crf.py) runs a small regularization grid.
- [10_build_gazetteer.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/10_build_gazetteer.py) builds a gazetteer from ATT&CK data.
- [11_breakdown.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/11_breakdown.py) supports model and label error analysis.

The phase notes report that CRF feature engineering improved the original all-sentence-training dev F1 from 0.192 to 0.200 for the entities layer and from 0.076 to 0.124 for the action layer. These values come from the phase documentation; they use an earlier experiment setup.

## 8. Relevance experiment and model comparison

An audit checked annotation offsets, BIO alignment, relevance labels, and report-level splits:

- All 15,487 raw annotated spans matched their recorded text and offsets.
- All annotated entity spans were in relevant sentences.
- Token-to-text offsets and BIO transitions validated across train/dev/test for both current layers.
- Train, dev, and test had no report overlap; all 100 reports were assigned.
- A heuristic search found known entity strings in many sentences marked irrelevant. These are candidates for unlabeled mentions—not verified labeling errors—but they suggest that treating every irrelevant sentence as a negative NER example may suppress recall.

The experiment in [12_relevant_only.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/12_relevant_only.py) compares relevant-only NER training, evaluates all dev sentences and relevant-only dev sentences, and tests TF-IDF Logistic Regression and LinearSVC as relevance classifiers. The test split is not loaded by that script.

For the **entities layer**, relevant-only dev F1 ranged from 0.264 for BiLSTM-CRF to 0.357 for spaCy. For the **action layer**, it ranged from 0.085 for token Logistic Regression to 0.212 for MEMM. These are exact-span seqeval scores from one dev split containing 10 reports; differences may be noisy.

The relevance classifiers had dev F1 of 0.679 for Logistic Regression and 0.685 for LinearSVC. Applied before the CRF, they modestly improved all-dev F1 over unfiltered CRF output in both layers, but their false negatives limit the two-stage system.

## 9. Completed Milestones (Phase 5: M0, M1, M2)

The project has completed the hygiene, leakage audit, and evaluation harness milestones of Phase 5:

1. **Milestone 0 (Baseline Lock & Hygiene)**:
   - Pinned core dependencies in `requirements.txt` and deep learning dependencies in `requirements-optional.txt`.
   - Populated `.gitignore` and untracked `.pyc` files while keeping all datasets tracked.
   - Recorded unedited baseline outputs in `results/baseline_before_changes.txt`.
   - Published learning note `docs/learning/M0_baseline_hygiene.md`.
2. **Milestone 1 (Leakage & Data Audit)**:
   - Audited gazetteer (`notebooks/10_build_gazetteer.py`): eliminated 13 leaked terms (`scilc.exe`, `telegram`, `active directory`, `plc`, `domain controller`, `github`, `svchost.exe`, `c2 server`, etc.), separated 57 ATT&CK campaigns into `CAMPAIGN`, added length/stop-word filters, and recorded term provenance in `data/external/gazetteer_provenance.json`.
   - Measured leakage impact (`results/gazetteer_leak_check.md`): removing `scilc.exe` dropped CRF `MALWARE_TOOL` F1 by 0.013, while MEMM improved by +0.008 F1.
   - Enforced test split lock: `load_smah` and `load_bio` refuse `test` without `CTI_ALLOW_TEST=1`.
   - Audited irrelevant sentences: identified 32,270 unlabelled entity mentions and exported a 100-mention sample to `results/unlabelled_mentions_sample.csv`.
   - Published learning note `docs/learning/M1_leakage_and_data_audit.md`.
3. **Milestone 2 (Evaluation Harness)**:
   - Built `src/cti/eval/`: exact seqeval (default and strict IOB2), relaxed span overlap, token-level macro/micro F1, 5-class error taxonomy, report-grouped K-fold cross-validation (`cv.py`), cluster bootstrapping (`bootstrap.py`), and reporting utilities (`report.py`).
   - Expanded unit test coverage from 11 to 26 passing unit tests in `tests/`.
   - Refactored `notebooks/07_compare.py` and `notebooks/12_relevant_only.py`.
   - Documented protocol in `docs/paper_protocol.md` and authored `docs/learning/M2_evaluation_harness.md`.
   - Produced baseline evaluation reports: `results/crf_baseline_m2_crf_tagger_entities.csv` and `.md`.

## 10. Milestone 3: CRF Tuning (Completed for Entities Layer)

With the evaluation harness and leakage-free dataset locked, Milestone 3 executed a 4-stage systematic tuning workflow:
1. **Feature extraction caching**: Built `src/cti/models/feature_cache.py` using `joblib` disk caching, slashing feature computation time from minutes to milliseconds.
2. **Feature ablation and model extension**: Extended `_sentence_features` and `CRFTagger` in `src/cti/models/crf.py` to support granular ablation switches (`window`, `use_ortho`, `use_affixes`, `use_ioc_patterns`, `use_gazetteer`, `use_bigrams`, `use_triggers`, `doc_context`, `word_clusters`) and exposed `algorithm`, `c1`, `c2`, `min_freq`, and `all_possible_transitions`.
3. **Stage 1 (Coarse Random Search - `results/crf_tuning/stage1_coarse_search_entities.csv`)**: Evaluated 30 parameter draws across 5 optimization algorithms. Key finding: stronger L1 regularization ($c_1 \approx 1.8 - 3.0$) and frequency filtering (`min_freq` $\ge 3$) are essential to prevent severe overfitting on 93k sparse features.
4. **Stage 2 (5-Fold Report-Grouped CV - `results/crf_tuning/stage2_cv_top5_entities.csv`)**: Evaluated the top 5 candidates across 5 document-grouped folds (80 reports: 70 train + 10 dev). Champion selected:
   - **Algorithm**: `lbfgs`
   - **$c_1$**: `1.843`, **$c_2$**: `0.137`
   - **`min_freq`**: `3`
   - **`all_possible_transitions`**: `True`
   - **5-Fold CV Exact F1**: **0.219 ± 0.036** (Relaxed F1: **0.310 ± 0.034**, Token Macro: **0.449 ± 0.025**).
5. **Stage 3 (Feature Ablation - `results/crf_tuning/stage3_ablation_entities.csv`)**: 5-fold CV across 9 feature conditions proved:
   - `no_gazetteer`: $\Delta = -0.0135 \rightarrow$ **KEEP (Essential)** ($+1.35\%$ F1 gain).
   - `no_orthography`: $\Delta = -0.0114 \rightarrow$ **KEEP (Essential)** ($+1.14\%$ F1 gain).
   - `no_affixes`: $\Delta = -0.0102 \rightarrow$ **KEEP (Essential)** ($+1.02\%$ F1 gain).
   - `window_1`: $\Delta = -0.0070 \rightarrow$ **KEEP ($w=\pm 2$ is required)**.
   - `window_3`: $\Delta = -0.0045 \rightarrow$ **REJECT** (slows training by 50% without benefit).
   - `no_ioc_patterns`: $\Delta = +0.0029 \rightarrow$ **NEUTRAL** (safely managed by dedicated IOC extractor).
6. **Stage 4 (New Features - `results/crf_tuning/stage4_new_features_entities.csv`)**:
   - `+doc_consistency`: $\Delta = -0.0064 \rightarrow$ **REJECT** (over-weights common repetitive non-entity words like 'attack', 'system').
   - `+word_clusters`: $\Delta = -0.0029 \rightarrow$ **NEUTRAL** (unsupervised 50 KMeans clusters).
7. **Educational Documentation**: Authored `docs/learning/M3_crf_tuning.md` covering linear-chain CRF theory, sequence modeling vs token classification, the 4-stage tuning rationale, empirical results, and model swapping via configuration.



