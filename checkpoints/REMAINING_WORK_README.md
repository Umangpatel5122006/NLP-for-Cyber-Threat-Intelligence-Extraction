# NLP for Cyber Threat Intelligence Extraction - Status Map and Remaining Work

Last synced: 2026-10-29

**Project goal (unchanged):** an NLP system that automatically extracts **cyber threats, attack techniques, vulnerabilities, affected systems and indicators of compromise (IOCs)** from security reports and threat-intelligence feeds.

**Professor's constraints:** no LLM, no direct LLM API, simpler ML + statistical NLP only. Every component stays modular (registry + `configs/config.yaml`) so it can be swapped for a newer model without touching other code. Language flexibility for Hindi, Marathi and Gujarati is built in as we go.

---

## 1. Where we are

| Phase | What it covers | Status | Evidence |
|---|---|---|---|
| P0 | Setup, plug-in architecture | **Done** | `core/base.py`, `registry.py`, `pipeline.py`, `config.yaml`, editable install |
| P1 | Data and labels | **Done** | SMAH-CTI + official split, `labels.yaml`, `load_smah()` |
| P2 | Preprocessing | **Done** | cleaner, defanger, language detector, sentence splitter, spaCy tokenizer |
| P3 | IOC extraction | **Done** | `ioc_finder` in pipeline, `ioc_regex` as fallback |
| P4 | NER training data (BIO) | **Done** | two layers (`entities`, `action`), span sanitation, 99-100% retention, audit passed |
| P5 | Statistical NER | **Mostly done** | CRF, MEMM, token LogReg, spaCy NER, BiLSTM-CRF, DistilBERT compared on dev |
| P5b | Relevance filter (new) | **Experimental** | TF-IDF LogReg/LinearSVC in a notebook only |
| P6 | Attack-technique classifier | Not started | labels exist in SMAH-CTI (`tactic`, `techniques`) |
| P6b | Vulnerabilities and affected systems (new) | Not started | no gold labels in SMAH-CTI |
| P7 | Relation extraction | Not started | |
| P8 | Full pipeline integration | **Partial** | preprocessing + IOC only; NER, relevance, techniques not in the config pipeline |
| P9 | Evaluation | **Partial** | exact-span seqeval on dev only; test split untouched |
| P10 | Output, demo, feed ingestion | Not started | |
| P11 | Upgrade guide, report, demo | Not started | |

**Exact position:** inside **P5**. The relevant-only comparison is finished. The next step is **P5 Step 8: verify how the paper scores NER and broaden our metrics.**

---

## 2. Latest results (dev set, exact-span seqeval)

Dev has only 10 reports, so differences of a few points are noise. The test split has not been used.

**Entities layer** (THREAT_ACTOR, MALWARE_TOOL, INFRASTRUCTURE), trained on relevant sentences only:

| Model | All-dev F1 | Relevant-dev F1 |
|---|---:|---:|
| CRF | 0.262 | 0.344 |
| MEMM | 0.254 | 0.352 |
| Token LogReg | 0.193 | 0.295 |
| spaCy NER | 0.249 | 0.357 |
| BiLSTM-CRF | 0.196 | 0.264 |

**Action layer**, relevant-dev F1: CRF 0.186, MEMM 0.212, spaCy 0.204, BiLSTM-CRF 0.186, token LogReg 0.085.

**Other results:**
- Earlier all-sentence training: CRF 0.200, DistilBERT 0.245 (2.5 h on CPU).
- Training on relevant sentences only lifted CRF all-dev F1 from 0.200 to 0.262 and recall from 0.137 to 0.287.
- Relevance classifiers reach dev F1 of about 0.68. Filtering before the CRF gives all-dev F1 of 0.274-0.278.
- The SMAH-CTI paper reports NER baseline F1 of roughly 74-81. **The gap is still about 40 points** and its cause is not yet explained.

---

## 3. Open question that drives the next work

Why are we far below 74-81? Candidate causes, to be tested in this order:

1. **Different scoring** (partial or token-level instead of exact match, or relevant sentences only).
2. **Unlabeled mentions** in sentences marked irrelevant, which teach the model that real entities are "O".
3. **Span-boundary conventions** in the annotations (exact match punishes small boundary differences).
4. **Model strength** (the paper's baselines may be fine-tuned transformers).
5. **Dev noise** (10 reports).

Do not tune further until causes 1-3 are checked.

---

## 4. Remaining work in detail

### Stage A - Finish P5: NER quality and evaluation

- [ ] **A1. Verify the paper's evaluation protocol.** Record: exact vs partial vs token-level; all sentences vs relevant only; micro vs macro; which models. Save as `docs/paper_protocol.md`.
- [ ] **A2. Broaden metrics.** One script that reports, per model and layer: exact-span, relaxed-overlap and token-level P/R/F1; per label; per source (aptnotes, crowdstrike, mandiant, mitre, sentinel). Save results to `results/*.csv`.
- [ ] **A3. Error taxonomy.** Count missed entities, spurious predictions, boundary errors and wrong-type errors. This shows which fix pays most.
- [ ] **A4. Training-set policy experiment.** Compare: relevant only; all sentences; relevant plus a sampled share of irrelevant sentences. Audit the irrelevant-sentence mentions found by the heuristic search (check about 100 by hand) and decide whether to mask or drop them.
- [ ] **A5. Context features.** Add `context_before`, sentence position, source name, and the relevance probability as features.
- [ ] **A6. CRF/MEMM feature upgrades** (all allowed, no LLM):
  - spaCy POS, lemma and, if it installs cleanly on Python 3.13, a small English pipeline
  - ioc_finder output as features
  - gazetteer partial and first-token alias matching
  - word clusters or static embeddings (gensim word2vec/fastText trained on the SMAH-CTI and ATT&CK text)
- [ ] **A7. Decision thresholds.** Use `predict_marginals` to trade precision for recall per label, tuned on dev.
- [ ] **A8. Split-layer experiment.** Three taggers (one each for THREAT_ACTOR, MALWARE_TOOL, INFRASTRUCTURE) plus a merge policy for overlaps. Needs `labels.yaml` layers updated and the BIO files rebuilt. Keep it only if it beats the single entity layer.
- [ ] **A9. Rules and post-processing.** Boundary normalization, generic mentions ("the threat actor", "the adversary"), file-name rules (decide MALWARE_TOOL vs INFRASTRUCTURE consistently).
- [ ] **A10. Ensemble.** Token- or span-level voting across CRF, MEMM and spaCy. Use stacking only if voting helps.
- [ ] **A11. Reduce dev noise.** Report mean and spread over several seeds, and a report-grouped cross-validation over train + dev.
- [ ] **A12. Action layer.** Repeat A4-A10 for `ACTION` (the weakest layer).
- [ ] **A13. Optional encoder upgrade.** Fine-tune a BERT-style model on relevant sentences only, with dev-based early stopping. Warn about CPU time (about 2.5 h for 3 epochs) and run a small sample first. This is a separate, clearly labeled module; confirm with the professor before reporting it as a project model. Check candidate model names on the Hugging Face Hub before use.

**Exit criteria:** best configuration per layer chosen on dev, a metrics table with all three scoring modes, and a written explanation of the gap to the paper.

### Stage B - P5b: relevance filter as a pipeline module

- [ ] **B1.** Move the TF-IDF classifier out of the notebook into `src/cti/models/relevance.py` (registered module, `fit`/`predict`, saved with joblib).
- [ ] **B2.** Add features: word + character n-grams, `context_before`, source name.
- [ ] **B3.** Tune the decision threshold for **recall**, because every missed relevant sentence is lost for the later stages.
- [ ] **B4.** Evaluate sentence-level P/R/F1 and the downstream effect on NER and technique mapping.
- [ ] **B5.** Alternatives to record for P11: Naive Bayes, gradient boosting, sentence embeddings + LogReg, fine-tuned encoder.

### Stage C - P6: attack-technique classifier

Matches the "attack techniques" part of the project title.

- [ ] **C1. Data analysis.** Label counts for tactics and techniques; 138 sentences have tactics without techniques; 696 sentences are multi-tactic and 565 multi-technique; many techniques are rare.
- [ ] **C2. Tactic classifier first** (about 14 labels): TF-IDF (word + char) with One-vs-Rest LogReg and LinearSVC.
- [ ] **C3. Technique classifier**: same family, plus a hierarchical option (tactic first, then techniques within it) and a fallback to the tactic when a technique is too rare.
- [ ] **C4. Use ATT&CK knowledge.** Technique names and descriptions from `enterprise-attack.json` as a TF-IDF similarity baseline for rare techniques.
- [ ] **C5. Features.** `context_before`, extracted ACTION spans, entity types present.
- [ ] **C6. Metrics.** Micro/macro F1, per-tactic F1, top-k accuracy, results on relevant sentences vs end-to-end after the relevance filter.
- [ ] **C7. Register `technique_classifier`**: writes ATT&CK IDs and names into `doc.metadata["techniques"]` per sentence.
- [ ] **C8. Alternatives for P11:** Naive Bayes, Random Forest, kNN, classifier chains, sentence embeddings + LogReg, encoder fine-tuning.

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
- [ ] **E2. Rule baseline:** same-sentence co-occurrence of two entities plus trigger verbs ("uses", "deploys", "exploits", "targets"); spaCy dependency patterns if a parser model is available.
- [ ] **E3. Classical ML:** LogReg on entity-pair features (entity types, words between, distance, trigger words).
- [ ] **E4. Validation:** check actor-malware pairs against ATT&CK "uses" relationships (distant supervision) and hand-check a sample of 100 relations, since SMAH-CTI has no relation gold.
- [ ] **E5. Register `relation_extractor`** writing to `doc.metadata["relations"]`.

### Stage F - P8: full pipeline integration

- [ ] **F1. Model persistence.** `train.py` CLI that trains each module and saves it under `models/` (joblib for CRF/MEMM/relevance, `to_disk` for spaCy) with a manifest: date, data version, dev metrics.
- [ ] **F2. NER modules in the pipeline:** `ner_entities` and `ner_action` wrap trained taggers: sentence split → tokenize → predict → BIO to character spans → `Entity` with confidence from marginals.
- [ ] **F3. Offset mapping.** Defanging changes text length, so keep a map from defanged positions back to the original report text. This is needed for highlighting and evidence in the output.
- [ ] **F4. Merge policy.** Resolve overlaps between IOC entities and NER entities (domain inside URL or email; INFRASTRUCTURE vs IOC), deduplicate, normalize (lowercase hashes, canonical domains), keep source and confidence.
- [ ] **F5. Final pipeline order:** cleaner → defanger → language detector → sentence splitter → ioc_finder → relevance filter → NER (entities, action) → system/vulnerability taggers → technique classifier → relation extractor → merger.
- [ ] **F6. Language routing.** English runs the full pipeline. For Hindi, Marathi and Gujarati: IOC extraction works as is (indicators are language-independent); for the rest, choose and test one strategy: multilingual encoder fine-tuned on English data and tested on a small hand-labeled sample, or an offline translation step, or an explicit "IOC-only" fallback with a warning. Create about 30 test sentences per language.
- [ ] **F7. Tests.** Unit test per new module; one end-to-end test on a short sample report; regression test that fails if dev F1 drops.

### Stage G - P10: outputs and input feeds

- [ ] **G1. Result schema (JSON):** entities with character positions, label, confidence, source module; IOCs; techniques with ATT&CK IDs; relations; metadata (language, model versions).
- [ ] **G2. STIX 2.1 export** with the `stix2` library: indicator, malware, tool, intrusion-set, vulnerability, attack-pattern, relationship, report. Validate the bundle.
- [ ] **G3. Input adapters.** Plain text, HTML (trafilatura or BeautifulSoup), PDF (pdfplumber or PyMuPDF), and RSS/advisory feeds, since the project covers threat-intelligence feeds.
- [ ] **G4. Interfaces.** CLI (`analyze report.txt`), a small FastAPI endpoint, and a Streamlit demo that highlights extracted entities in the report text.
- [ ] **G5. Real-report check.** Run the full system on 3-5 public reports that are **not** in SMAH-CTI and review the output by hand.

### Stage H - P9: final evaluation (done last)

- [ ] **H1.** Freeze all configurations using dev only.
- [ ] **H2.** Run the **test split once** and report: NER per layer and label (three scoring modes), relevance, technique classification, IOC extraction (on the hand-labeled sample, since SMAH-CTI has no IOC gold), SYSTEM/VULNERABILITY (on the small gold set), relations (sampled).
- [ ] **H3.** Ablations: gazetteer on/off, relevance filter on/off, context features on/off.
- [ ] **H4.** Confidence intervals by bootstrapping over reports.
- [ ] **H5.** A section explaining the comparison with the paper's numbers, based on A1.

### Stage I - P11: upgrade guide, report and demo

- [ ] **I1. Swap-in table** for every module: current model, modern alternative, how to swap (one config line), cost and licensing.

| Module | Now | Modern swap-ins |
|---|---|---|
| NER | CRF / MEMM / spaCy | BiLSTM-CRF, BERT-style encoders (DistilBERT, security-domain encoders such as SecBERT or CyBERT: verify on the Hub), span-based encoder NER |
| Relevance | TF-IDF + LogReg | sentence embeddings + LogReg, fine-tuned encoder |
| Techniques | TF-IDF + LogReg | embeddings + LogReg, encoder fine-tuning; MITRE TRAM is a real-world reference (check its current design) |
| IOC | ioc-finder | iocextract, iocsearcher (verify), custom rules |
| Language ID | script heuristic | fastText language-ID, lingua |
| Sentence splitting | regex | NLTK Punkt, wtpsplit (multilingual neural) |
| Relations | patterns / LogReg | encoder-based relation classifiers |
| Output | JSON / STIX 2.1 | TAXII server, MISP or OpenCTI connectors, SIEM export |

- [ ] **I2. Production notes:** Docker packaging, model registry/versioning (for example MLflow), monitoring for drift, and a human-review loop that feeds corrected labels back into training.
- [ ] **I3. Note on industry practice:** many current systems use LLM-based extraction. That approach is excluded from this project by the professor's rule, and the report should say so.
- [ ] **I4. Final report and demo script:** architecture diagram, dataset and license note (CC BY-NC 4.0, non-commercial), results tables, limitations.

### Stage J - Engineering hygiene (do alongside the other stages)

- [ ] **J1. Commit the uncommitted work now:** `README.md`, `tests/test_models_preprocessing.py`, `src/cti/models/memm.py`, `notebooks/12_relevant_only.py`.
- [ ] **J2. Dependencies.** Create `requirements.txt` (or optional dependency groups in `pyproject.toml`) for sklearn-crfsuite, spaCy, ioc-finder, seqeval, pyyaml, and optional torch/transformers.
- [ ] **J3. `.gitignore`** for `venv/`, `data/raw`, `data/external`, `models/`, caches.
- [ ] **J4. Reproducibility:** fixed seeds, results saved in `results/`, one script that regenerates every table.
- [ ] **J5. Keep the test split locked** until Stage H.

---

## 5. Immediate next steps (in order)

1. **A1** Read the paper's evaluation section and record the protocol.
2. **A2** Add relaxed and token-level metrics with per-label and per-source output.
3. **J1** Commit the current work.
4. **A4** Training-set policy experiment and audit of unlabeled mentions.
5. **B1** Move the relevance classifier into a registered module.

## 6. Definition of done (maps to the project description)

- [ ] **Threats** (actors, malware, tools): NER in the pipeline, scored on dev and test
- [ ] **Attack techniques:** ATT&CK tactic and technique classifier in the pipeline
- [ ] **Vulnerabilities:** CVE IDs, enrichment and vulnerability descriptions extracted
- [ ] **Affected systems:** system/software tagger with a small gold evaluation
- [ ] **IOCs:** extracted from reports and feeds, defanging handled, STIX output
- [ ] **Reports and feeds** accepted as input (text, HTML, PDF, RSS)
- [ ] **No LLM**, modular swap-ins documented, language flexibility demonstrated
