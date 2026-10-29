# Project progress, in order

## 1. Project goal and structure

The project aims to extract threat-related entities, actions, and indicators of compromise (IOCs) from cybersecurity reports. Its core design is modular: components register under names, and a pipeline builds them from configuration so modules can be changed without rewriting the pipeline.

- [pyproject.toml](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/pyproject.toml) sets up the installable Python package.
- [base.py: `Entity`, `Document`, `BaseModule`](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/core/base.py) defines the common document and pipeline-module contract.
- [registry.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/core/registry.py) registers components by name and creates them.
- [pipeline.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/core/pipeline.py) loads registered modules and runs the configured pipeline.

**Current pipeline order**, from [config.yaml](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/configs/config.yaml), is: cleaner → defanger → language detector → sentence splitter → IOC finder. NER models have their own shared tagger interface, but NER is currently evaluated through experiment scripts rather than added to this configured document pipeline.

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

**Important boundary detail:** defanging changes the text. Pipeline positions after defanging refer to the changed text, not the original text; there is no general offset-mapping mechanism yet. The dataset’s BIO tags, however, are built from the original annotated sentences, not from text run through the pipeline.

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

## 9. What remains unfinished

The project has working data preparation, preprocessing, IOC extraction, BIO generation, multiple model implementations, and dev-only experiments. Several evaluation or integration tasks remain:

1. **Confirm comparison with the SMAH-CTI paper.** The project notes report the paper’s baseline at roughly 74–81 F1, but the exact scoring and evaluation setup have not yet been verified against the current results.
2. **Broaden metrics.** The latest benchmark reports exact-span seqeval metrics and per-label reports. A comparable partial-overlap and token-level evaluation, plus metrics by data source, is not yet included in that benchmark.
3. **Integrate relevance filtering into the app pipeline.** The TF-IDF classifiers are currently experimental code in the notebook, not registered pipeline modules; the notebook does not use `context_before`.
4. **Test three separate entity layers.** `THREAT_ACTOR`, `MALWARE_TOOL`, and `INFRASTRUCTURE` are still combined in `entities`. Separate taggers might help or might introduce conflicts; this has not been measured.
5. **Improve entity/action performance.** The reported best conditional dev results are still modest, especially for `ACTION` and `INFRASTRUCTURE`.
6. **Keep the test split untouched** until model design and dev comparisons are complete; only then run one final test evaluation.

## 10. Current working-tree note

At the time of this summary, the working tree has uncommitted changes in [README.md](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/README.md) and [test_models_preprocessing.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/tests/test_models_preprocessing.py), plus new files [memm.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/src/cti/models/memm.py) and [12_relevant_only.py](/E:/STUDY/college-projects/NLP-for-Cyber-Threat-Intelligence-Extraction/notebooks/12_relevant_only.py). The focused regression suite previously passed 11 tests. No final evaluation on the test split has been reported.
