# NLP for Cyber Threat Intelligence Extraction

An NLP-powered system that extracts cyber threats, attack techniques, vulnerabilities, affected systems, and indicators of compromise (IOCs) from unstructured security reports and threat intelligence feeds.

---

## Architecture & Pipeline

The pipeline follows a modular, plug-in architecture configured via `configs/config.yaml`:
1. **Cleaner (`cleaner`)**: Unicode normalization (NFC), strips non-essential invisible characters (`\u200b`, `\ufeff`) while preserving Indic characters for cross-lingual support.
2. **Defanger (`defanger`)**: Normalizes obfuscated indicators (e.g., `hxxps://`, `evil[.]com`, `[:]`, `[at]`) back to canonical form while preserving the original raw text in `doc.metadata["text_before_defang"]`.
3. **Language Detector (`language_detector`)**: Identifies document script and language (`en`, `hi`, `mr`, `gu`).
4. **Sentence Splitter (`sentence_splitter`)**: Sentence boundary detection aware of IP addresses, domain names, CTI abbreviations (e.g., `e.g.`, `i.e.`), and the Devanagari danda.
5. **IOC Extractor (`ioc_finder` / `ioc_regex`)**: Rule-based indicator extraction for IPs, domains, URLs, hashes, CVEs, and emails.
6. **Statistical NER Taggers**: Models implement the shared `BaseTagger` interface and can be selected by registry name. Current options include `crf_tagger`, `memm_tagger`, `token_logreg`, `spacy_ner`, and `bilstm_crf`. Each is trained on token-aligned BIO tags across two specialized layers:
   - `entities` layer: `THREAT_ACTOR`, `MALWARE_TOOL`, `INFRASTRUCTURE`
   - `action` layer: `ACTION`

---

## Relevant-Only Model Comparison (Dev Set)

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
| spaCy NER (`spacy_ner`) | 0.199 | 0.332 | 0.249 | 0.386 | 0.332 | **0.357** |
| BiLSTM-CRF (`bilstm_crf`) | 0.182 | 0.212 | 0.196 | 0.350 | 0.212 | 0.264 |

Per-label relevant-dev F1:

| Model | THREAT_ACTOR | MALWARE_TOOL | INFRASTRUCTURE |
|---|---:|---:|---:|
| CRF | 0.485 | 0.289 | 0.312 |
| MEMM | **0.507** | 0.306 | 0.311 |
| Token Logistic Regression | 0.421 | **0.309** | 0.226 |
| spaCy NER | 0.522 | 0.335 | 0.285 |
| BiLSTM-CRF | 0.455 | 0.174 | 0.233 |

The TF-IDF relevance classifiers scored positive-class (relevant) F1 of 0.679
for Logistic Regression and 0.685 for LinearSVC. Filtering CRF output with
those predictions yielded end-to-end all-dev F1 of 0.278 and 0.274,
respectively. This is only a modest improvement over the unfiltered CRF's
0.262 all-dev F1: missed relevant sentences limit the two-stage pipeline.
These results use one dev split of 10 reports and are not a stable estimate
of generalization. The test split was not used.

Run the full comparison from the repository root:

```powershell
.\venv\Scripts\python.exe notebooks\12_relevant_only.py
```

Choose a layer or model subset without editing the script:

```powershell
.\venv\Scripts\python.exe notebooks\12_relevant_only.py --layer action
.\venv\Scripts\python.exe notebooks\12_relevant_only.py --models crf_tagger,memm_tagger
```

The BERT-style encoder remains an optional, separate upgrade; it is excluded
from this CPU-friendly comparison because the earlier DistilBERT run took
about 2.5 hours.

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
| spaCy NER (`spacy_ner`) | 0.096 | 0.202 | 0.130 | 0.205 | 0.202 | 0.204 |
| BiLSTM-CRF (`bilstm_crf`) | 0.122 | 0.161 | 0.139 | 0.221 | 0.161 | 0.186 |

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
.\venv\Scripts\python.exe notebooks\12_relevant_only.py --layer entities
.\venv\Scripts\python.exe notebooks\12_relevant_only.py --layer action
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
current CRF, MEMM, spaCy, and other models on relevant dev sentences.

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
# Run unit test suite (26 tests)
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
