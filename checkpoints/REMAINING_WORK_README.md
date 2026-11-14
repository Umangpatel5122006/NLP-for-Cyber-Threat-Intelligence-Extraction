# NLP for Cyber Threat Intelligence Extraction - Status Map and Remaining Work

Last synced: 2026-11-15

**Project goal (unchanged):** an NLP system that automatically extracts **cyber threats, attack techniques, vulnerabilities, affected systems and indicators of compromise (IOCs)** from security reports and threat-intelligence feeds.

**Confirmed methods constraint:** classical NLP and classical ML only. No deep learning: BERT/transformers, BiLSTMs, neural embeddings, trained spaCy NER/parsers, and LLMs are excluded. spaCy's blank tokenizer and rule-based matcher remain allowed. Older checkpoint suggestions for neural upgrades are superseded. See `AGENTS.md` and `checkpoints/CLASSICAL_SCOPE.md`.

---

## 1. Where we are

| Phase | What it covers | Status | Evidence |
|---|---|---|---|
| P0 | Setup, plug-in architecture | **Done** | `core/base.py`, `registry.py`, `pipeline.py`, `config.yaml`, editable install |
| P1 | Data and labels | **Done** | SMAH-CTI + official split, `labels.yaml`, `load_smah()` |
| P2 | Preprocessing | **Done** | cleaner, defanger, language detector, sentence splitter, spaCy tokenizer |
| P3 | IOC extraction | **Done** | `ioc_finder` in pipeline, `ioc_regex` as fallback |
| P4 | NER training data (BIO) | **Done** | two layers (`entities`, `action`), span sanitation, 99-100% retention, audit passed |
| P5 | Statistical NER | **Saved classical baselines integrated** | CRF entity/action artifacts, train/dev CLI and reports; MEMM and token LogReg persistence tested |
| P5b | Relevance filter | **Reusable module complete** | saved TF-IDF + LogReg, train/dev CLI, opt-in document pipeline; threshold work remains |
| P6 | Attack-technique classifier | Not started | labels exist in SMAH-CTI (`tactic`, `techniques`) |
| P6b | Vulnerabilities and affected systems (new) | Not started | no gold labels in SMAH-CTI |
| P7 | Relation extraction | Not started | |
| P8 | Full pipeline integration | **Partial** | preprocessing + IOC + saved relevance/NER; original offsets, merging and techniques remain |
| P9 | Evaluation | **Partial** | corrected metrics version 2, grouped validation utilities; official test reserved |
| P10 | Output, demo, feed ingestion | **CLI and JSON baseline** | `cti.analyze`, sample report/output; frontend, exports and feeds remain |
| P11 | Upgrade guide, report, demo | Not started | |

**Current position:** saved entity/action taggers now run in the document pipeline,
with corrected dev diagnostics and identical predictions after reload. All 78
tests pass in WSL. The preceding scope commit passed 62 tests in GitHub Actions;
the updated NER workflow awaits its remote run. See [NER_INTEGRATION.md](NER_INTEGRATION.md).
Original-text offset mapping is the next reviewable integration step.

---

## 2. Latest results (dev set, exact-span seqeval)

Dev has only 10 reports, so differences of a few points are noise. The test split has not been used.

**Entities layer** (THREAT_ACTOR, MALWARE_TOOL, INFRASTRUCTURE), trained on relevant sentences only:

| Model | All-dev F1 | Relevant-dev F1 |
|---|---:|---:|
| CRF | 0.262 | 0.344 |
| MEMM | 0.254 | 0.352 |
| Token LogReg | 0.193 | 0.295 |

**Action layer**, historical relevant-dev F1: CRF 0.186, MEMM 0.212, token LogReg 0.085. Excluded neural results are preserved in `docs/history/neural_comparisons.md`.

**Other results:**
- Earlier all-sentence CRF training: dev exact F1 0.200.
- Training on relevant sentences only lifted CRF all-dev F1 from 0.200 to 0.262 and recall from 0.137 to 0.287.
- Relevance classifiers reach dev F1 of about 0.68. Filtering before the CRF gives all-dev F1 of 0.274-0.278.
- The verified released paper reports strict character-span NER F1 around 0.14-0.22, not the previously assumed 74-81. Direct comparison still requires matching split, label layers, and original span conventions. See `docs/paper_protocol.md`.

---

## 3. Questions that drive the next work

The supposed 40-point gap was based on an unsupported benchmark assumption.
Focus next on exact-span errors, incomplete annotations, sentence relevance
recall, source variation, and stable report-grouped comparisons of the allowed
classical models. Historical token/relaxed figures need recomputation under
metrics version 2. Neural model upgrades are outside project scope.

---

## 4. Remaining work in detail

### Stage A - Finish P5: NER quality and evaluation

- [x] **A1. Verify the paper's evaluation protocol.** Recorded: exact vs partial vs token-level; all sentences vs relevant only; micro vs macro; which models. Saved as `docs/paper_protocol.md`.
- [x] **A2. Broaden metrics.** Multi-perspective evaluation in `src/cti/eval/metrics.py` and `report.py` reporting exact-span (default & strict IOB2), relaxed-overlap, and token-level P/R/F1 across 5 document sources. Saved to `results/*.csv`.
- [x] **A3. Error taxonomy.** 5-class error taxonomy implemented in `src/cti/eval/metrics.py`: exact matches, boundary errors, wrong-type errors, missed entities, and spurious predictions.
- [x] **A4. Training-set policy experiment & audit.** Audited unlabelled mentions in irrelevant sentences (32,270 mentions found; 100-mention sample exported in `results/unlabelled_mentions_sample.csv`). Audited and remediated 13 leaked gazetteer terms in M1.
- [ ] **A5. Context features.** Add `context_before`, sentence position, source name, and relevance probability as features in the configured document pipeline.
- [x] **A6. CRF feature upgrades & ablation.** Executed in M3 across 4 stages: coarse random search (`14_tune_crf_stage1.py`), 5-fold grouped CV (`15_tune_crf_stage2_cv.py`), 9-condition ablation (`16_crf_ablation.py`), and new feature exploration (`17_crf_new_features.py`).
- [ ] **A7. Decision thresholds.** Use `predict_marginals` to trade precision for recall per label, tuned on dev.
- [ ] **A8. Split-layer experiment.** Three taggers (one each for THREAT_ACTOR, MALWARE_TOOL, INFRASTRUCTURE) plus a merge policy for overlaps.
- [ ] **A9. Rules and post-processing.** Boundary normalization, generic mentions, file-name rules.
- [ ] **A10. Ensemble.** Token- or span-level voting across CRF, MEMM and token Logistic Regression.
- [x] **A11. Reduce dev noise.** Implemented 5-fold report-grouped cross-validation (`src/cti/eval/cv.py`) and cluster bootstrapping (`src/cti/eval/bootstrap.py`) over all 80 train+dev reports.
- [ ] **A12. Action layer.** Repeat tuning for `ACTION` layer.
- **A13. Excluded:** neural encoder upgrades are outside the confirmed scope.


**Exit criteria:** best configuration per layer chosen on dev, a metrics table with all three scoring modes, and a written explanation of the gap to the paper.

### Stage B - P5b: relevance filter as a pipeline module

- [x] **B1.** Reusable TF-IDF classifier in `src/cti/models/relevance.py`, registered document module, joblib persistence, train/dev CLI, and saved dev predictions. See `checkpoints/RELEVANCE_MODULE.md`. The old notebook remains as experiment history.
- [ ] **B2.** Add features: word + character n-grams, `context_before`, source name.
- [ ] **B3.** Tune the decision threshold for **recall**, because every missed relevant sentence is lost for the later stages.
- [ ] **B4.** Sentence-level dev P/R/F1 is recorded (0.6265 / 0.7419 / 0.6793 at threshold 0.5); downstream NER and technique effects still need evaluation.
- [ ] **B5.** Classical alternatives: Naive Bayes, LinearSVC, word/character n-grams, and context features.

### Stage C - P6: attack-technique classifier

Matches the "attack techniques" part of the project title.

- [ ] **C1. Data analysis.** Label counts for tactics and techniques; 138 sentences have tactics without techniques; 696 sentences are multi-tactic and 565 multi-technique; many techniques are rare.
- [ ] **C2. Tactic classifier first** (about 14 labels): TF-IDF (word + char) with One-vs-Rest LogReg and LinearSVC.
- [ ] **C3. Technique classifier**: same family, plus a hierarchical option (tactic first, then techniques within it) and a fallback to the tactic when a technique is too rare.
- [ ] **C4. Use ATT&CK knowledge.** Technique names and descriptions from `enterprise-attack.json` as a TF-IDF similarity baseline for rare techniques.
- [ ] **C5. Features.** `context_before`, extracted ACTION spans, entity types present.
- [ ] **C6. Metrics.** Micro/macro F1, per-tactic F1, top-k accuracy, results on relevant sentences vs end-to-end after the relevance filter.
- [ ] **C7. Register `technique_classifier`**: writes ATT&CK IDs and names into `doc.metadata["techniques"]` per sentence.
- [ ] **C8. Classical alternatives:** Naive Bayes, Random Forest, kNN, classifier chains, and TF-IDF similarity retrieval.

### Stage D - P6b: vulnerabilities and affected systems

Matches "vulnerabilities" and "affected systems" in the title. SMAH-CTI has no gold labels for them, so this stage needs its own data.

- [ ] **D1. Vulnerability identifiers.** CVE IDs already come from `ioc_finder`; normalize them (uppercase, deduplicate).
- [ ] **D2. CVE enrichment.** Offline NVD data (description, CVSS, affected products/CPE) and a CWE name list for vulnerability types ("command injection", "buffer overflow").
- [ ] **D3. Vulnerability-description tagger.** CWE-name gazetteer + patterns as a registered module; check the output against the INFRASTRUCTURE and ACTION spans that already describe vulnerabilities.
- [ ] **D4. Affected-system tagger.** Dictionary of vendors, products and operating systems from NVD CPE data, with fuzzy and alias matching; registered as `system_gazetteer`.
- [ ] **D5. Split INFRASTRUCTURE.** Use the dictionaries as weak supervision to separate software/system mentions from files and other infrastructure, then update `later_labels` (`SYSTEM`, `VULNERABILITY`) in `labels.yaml`.
- [ ] **D6. Small gold set.** Hand-label 100-200 sentences (for example with Label Studio or doccano) so SYSTEM and VULNERABILITY can be scored honestly.
- [ ] **D7. Alternative datasets to evaluate:** APTNER, DNRTI (check availability and licenses first); NVD text for vulnerability phrasing.

### Stage E - P7: relation extraction

- [ ] **E1. Define relation types:** actor-uses-malware/tool, malware-exploits-CVE, actor/malware-targets-system, indicator-associated-with-malware.
- [ ] **E2. Rule baseline:** generate same-sentence entity-pair candidates, then apply trigger verbs ("uses", "deploys", "exploits", "targets") and negation rules. Do not use a neural dependency parser.
- [ ] **E3. Classical ML:** LogReg on entity-pair features (entity types, words between, distance, trigger words).
- [ ] **E4. Validation:** check actor-malware pairs against ATT&CK "uses" relationships (distant supervision) and hand-check a sample of 100 relations, since SMAH-CTI has no relation gold.
- [ ] **E5. Register `relation_extractor`** writing to `doc.metadata["relations"]`.

### Stage F - P8: full pipeline integration

- [x] **F1. Model persistence.** Relevance and classical entity/action artifacts have manifests and train/dev reload checks. CRF, MEMM and token LogReg round trips are tested.
- [x] **F2. NER modules in the pipeline:** `ner_entities` and `ner_action` wrap saved taggers: sentence split → tokenize → predict → BIO to character spans. Confidence is explicitly unavailable (`null`); calibrated marginal confidence remains optional future work.
- [ ] **F3. Offset mapping.** Defanging changes text length, so keep a map from defanged positions back to the original report text. This is needed for highlighting and evidence in the output.
- [ ] **F4. Merge policy.** Resolve overlaps between IOC entities and NER entities (domain inside URL or email; INFRASTRUCTURE vs IOC), deduplicate, normalize (lowercase hashes, canonical domains), keep source and confidence.
- [ ] **F5. Final pipeline order:** cleaner → defanger → language detector → sentence splitter → ioc_finder → relevance filter → NER (entities, action) → system/vulnerability taggers → technique classifier → relation extractor → merger.
- [ ] **F6. Language routing.** Validate an explicit IOC-only fallback for Hindi, Marathi and Gujarati first. Further extraction needs tested language-specific dictionaries/rules or classical taggers with suitable annotated data. Neural translation/encoders are excluded. Create a small evaluation sample per language.
- [ ] **F7. Tests.** Unit test per new module; one end-to-end test on a short sample report; regression test that fails if dev F1 drops.

### Stage G - P10: outputs and input feeds

- [ ] **G1. Result schema (JSON):** entities with character positions, label, confidence, source module; IOCs; techniques with ATT&CK IDs; relations; metadata (language, model versions).
- [ ] **G2. STIX 2.1 export** with the `stix2` library: indicator, malware, tool, intrusion-set, vulnerability, attack-pattern, relationship, report. Validate the bundle.
- [ ] **G3. Input adapters.** Plain text, HTML (trafilatura or BeautifulSoup), PDF (pdfplumber or PyMuPDF), and RSS/advisory feeds, since the project covers threat-intelligence feeds.
- [ ] **G4. Interfaces.** `python -m cti.analyze --input report.txt` provides the CLI/JSON baseline. A small FastAPI endpoint and frontend highlighting remain; implement original-report offset mapping first.
- [ ] **G5. Real-report check.** Run the full system on 3-5 public reports that are **not** in SMAH-CTI and review the output by hand.

### Stage H - P9: final evaluation (done last)

- [ ] **H1.** Freeze all configurations using dev only.
- [ ] **H2.** Run the **test split once** and report: NER per layer and label (three scoring modes), relevance, technique classification, IOC extraction (on the hand-labeled sample, since SMAH-CTI has no IOC gold), SYSTEM/VULNERABILITY (on the small gold set), relations (sampled).
- [ ] **H3.** Ablations: gazetteer on/off, relevance filter on/off, context features on/off.
- [ ] **H4.** Confidence intervals by bootstrapping over reports.
- [ ] **H5.** A section explaining the comparison with the paper's numbers, based on A1.

### Stage I - P11: upgrade guide, report and demo

- [ ] **I1. Swap-in table** for every module: current method, allowed classical alternative, integration, cost and licensing.

| Module | Now | Allowed alternatives |
|---|---|---|
| NER | CRF / MEMM / token LogReg | HMM, structured perceptron, label-specific CRFs, dictionary/rule features |
| Relevance | TF-IDF + LogReg | character n-grams + LinearSVC, Naive Bayes, context features |
| Techniques | planned TF-IDF + linear models | classifier chains, hierarchical classifiers, lexical similarity |
| IOC | ioc-finder | existing custom regex and normalization rules |
| Language ID | script heuristic | character n-gram classical classifier, validated dictionaries |
| Sentence splitting | regex | rule refinement, classical statistical sentence segmentation |
| Relations | planned patterns / LogReg | entity-pair features + linear classifier, trigger/negation rules |
| Output | JSON / STIX 2.1 | TAXII server, MISP or OpenCTI connectors, SIEM export |

- [ ] **I2. Production notes:** Docker packaging, model registry/versioning (for example MLflow), monitoring for drift, and a human-review loop that feeds corrected labels back into training.
- [ ] **I3. Method scope:** explain that all deep learning, including trained spaCy NER and neural encoders, is excluded. Keep earlier neural results in historical records only.
- [ ] **I4. Final report and demo script:** architecture diagram, dataset and license note (CC BY-NC 4.0, non-commercial), results tables, limitations.

### Stage J - Engineering hygiene (do alongside the other stages)

- [ ] **J1. Commit the uncommitted work now:** `README.md`, `tests/test_models_preprocessing.py`, `src/cti/models/memm.py`, `notebooks/12_relevant_only.py`.
- [x] **J2. Dependencies.** Pinned core dependencies and metadata are present. The neural extra is removed; its former requirements are archived under `docs/history/`.
- [ ] **J3. `.gitignore`** for `venv/`, `data/raw`, `data/external`, `models/`, caches.
- [ ] **J4. Reproducibility:** fixed seeds, results saved in `results/`, one script that regenerates every table.
- [ ] **J5. Keep the test split locked** until Stage H.

---

## 5. Immediate next steps (in order)

1. Review and run the prepared classical CI workflow after the user's normal commit/push.
2. Record a full passing classical baseline in a runtime that loads the tokenizer.
3. Add saved CRF/MEMM entity/action inference to the document pipeline.
4. Regenerate dev diagnostics under metrics version 2 and compare relevance thresholds.
5. Implement TF-IDF + linear multi-label ATT&CK tactic/technique classifiers.

## 6. Definition of done (maps to the project description)

- [ ] **Threats** (actors, malware, tools): NER in the pipeline, scored on dev and test
- [ ] **Attack techniques:** ATT&CK tactic and technique classifier in the pipeline
- [ ] **Vulnerabilities:** CVE IDs, enrichment and vulnerability descriptions extracted
- [ ] **Affected systems:** system/software tagger with a small gold evaluation
- [ ] **IOCs:** extracted from reports and feeds, defanging handled, STIX output
- [ ] **Reports and feeds** accepted as input (text, HTML, PDF, RSS)
- [ ] **Classical NLP/ML only, no deep learning**, modular alternatives documented, language support evaluated honestly
