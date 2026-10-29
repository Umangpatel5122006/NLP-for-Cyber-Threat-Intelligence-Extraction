# Phase 3, Phase 4, and Phase 5 - IOC Extraction, BIO Alignment, and Statistical NER

Project: NLP for Cyber Threat Intelligence Extraction

---

## Phase 3 - IOC Extractor

**Goal:** find indicators of compromise (IP, DOMAIN, URL, HASH, CVE, EMAIL) with rule-based extraction.

### What we did
1. **Hand-written regex module** `src/cti/extractors/ioc_regex.py` (registered as `ioc_regex`)
   - Handles IP (validated 0-255), CVE, HASH (MD5/SHA-1/SHA-256). Kept as a fallback and as a learning step.
2. **Library swap** `src/cti/extractors/ioc_finder.py` (registered as `ioc_finder`)
   - Uses the `ioc-finder` library: IPv4/IPv6, domains, URLs, emails, CVEs, MD5/SHA-1/SHA-256/SHA-512.
   - Library gives values without positions, so the module locates each value in the text to build `Entity(start, end)`.
   - Swapped in by one line in `configs/config.yaml` (`ioc_finder` last in the pipeline).
3. **Real-data check** (`notebooks/04_ioc_check.py`) on SMAH-CTI dev:
   - Without defanging: 6.6% of `Infrastructure_Indicator` spans caught.
   - With defanger + ioc_finder: 36 of 412 (**8.7%**).

---

## Phase 4 - NER Training Data (BIO Tagging & Alignment Verification)

**Goal:** turn SMAH-CTI character spans into token-level BIO tags that models can accurately learn from.

### What we did
1. **Tokenizer** `src/cti/preprocessing/tokenizer.py`
   - Uses spaCy `spacy.blank(lang)` (tokenizer only, no model download); falls back to multilingual `xx`.
   - Returns `(token, start, end)`. Keeps IPs and domains whole; splits Hindi/Marathi/Gujarati words and the danda correctly.
2. **BIO Converter & Span Sanitation** `src/cti/core/bio.py`
   - Fixed raw annotation noise: over 21% of raw entity spans contained leading/trailing spaces, and hundreds ended with stray clause punctuation (`.`, `,`, `;`, `:`, quotes).
   - In `to_bio()`, edge whitespace and stray boundary punctuation are trimmed before token snapping (`char_span(alignment_mode="expand")`), preventing punctuation from becoming false entity tokens (`I-MALWARE_TOOL`, etc.).
   - Overlaps resolved via `filter_spans()`.
3. **Layer Separation**:
   - `entities` layer: THREAT_ACTOR, MALWARE_TOOL, INFRASTRUCTURE
   - `action` layer: ACTION
4. **Rebuilt Datasets** (`notebooks/05_build_bio.py`):
   - `data/processed/bio_en_{entities,action}_{train,dev,test}.json`

### Dataset Retention
| Split | Layer | Gold | Kept | Retention |
|---|---|---|---|---|
| train | entities | 5,957 | 5,914 | 99.3% |
| train | action | 4,470 | 4,453 | 99.6% |
| dev | entities | 912 | 903 | 99.0% |
| dev | action | 608 | 608 | 100.0% |
| test | entities | 2,072 | 2,054 | 99.1% |
| test | action | 1,468 | 1,453 | 99.0% |

---

## Phase 5 - Statistical NER (CRF Model Optimization)

**Goal:** optimize feature engineering, regularizations, and gazetteer lookups in `src/cti/models/crf.py` to boost precision, recall, and F1.

### Key Improvements in Statistical Model (`crf_tagger`)
1. **Sanitized Morphology & Affixes**:
   - Prefix/suffix features (1 to 4 chars) are computed on cleaned lowercase tokens rather than raw punctuation-bearing tokens, ensuring accurate morphological signals (e.g., `.exe`, `.dll`, `-ing`, `-tion`).
2. **High-Performance IPv6 Detection**:
   - Early character guard added (`':' in value and value.count(':') >= 2`) to eliminate 200,000+ slow exception catches, reducing feature extraction time from 160s+ down to ~7s.
3. **Enhanced Gazetteer & Positive-Evidence Clues**:
   - Extended `data/external/gazetteer.json` with base names (e.g., "Lazarus Group" -> "Lazarus"), ATT&CK campaigns, and infrastructure terms.
   - Fixed negative bias: gazetteer features only emit positive evidence (`"B"` / `"I"`). Eliminated constant `"O"` emission that previously penalized unknown entities with large negative weights (-2.19).
4. **Context Window & Boundary Alignment**:
   - Sentence boundaries `BOS` and `EOS` are set strictly at `i == 0` and `i == n - 1`.
   - Bidirectional word bigrams (`bigram` and `+1:bigram`) provide strong syntactic coupling with preceding and following context tokens.

### Performance Comparison (Dev Set)

#### Entities Layer (`THREAT_ACTOR`, `MALWARE_TOOL`, `INFRASTRUCTURE`)
| Metric | Baseline CRF | Updated CRF | Change |
|---|---|---|---|
| **Precision** | 0.365 | **0.370** | **+1.4%** |
| **Recall** | 0.131 | **0.137** | **+4.6%** |
| **F1 Score** | 0.192 | **0.200** | **+4.2%** |
| THREAT_ACTOR F1 | 0.309 | **0.327** | **+5.8%** |
| MALWARE_TOOL F1 | 0.193 | **0.195** | **+1.0%** |
| INFRASTRUCTURE F1 | 0.127 | **0.132** | **+3.9%** |

#### Action Layer (`ACTION`)
| Metric | Baseline CRF | Updated CRF | Relative Change |
|---|---|---|---|
| **Precision** | 0.148 | **0.224** | **+51.4%** |
| **Recall** | 0.051 | **0.086** | **+68.6%** |
| **F1 Score** | 0.076 | **0.124** | **+63.2%** |
