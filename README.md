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
6. **Statistical NER Tagger (`crf_tagger`)**: Sequence labeler using Conditional Random Fields (CRF) trained on token-aligned BIO tags across two specialized layers:
   - `entities` layer: `THREAT_ACTOR`, `MALWARE_TOOL`, `INFRASTRUCTURE`
   - `action` layer: `ACTION`

---

## NER Model Input & BIO Tagging

- Taggers in `src/cti/models` consume pre-tokenized sentences and aligned BIO tags via `fit(tokens, tags, dev_tokens=None, dev_tags=None)` and predict tag sequences via `predict(tokens)`.
- **Span Sanitization in `src/cti/core/bio.py`**: When converting character offsets to token-level BIO tags (`to_bio`), leading/trailing whitespace and stray sentence/clause boundary punctuation (`.`, `,`, `;`, `:`, quotes) are trimmed before snapping with `alignment_mode="expand"`. This prevents punctuation tokens from being incorrectly labeled as entities and ensures clean token-level boundaries.
- **Gazetteer Grounding**: Backed by MITRE ATT&CK (`data/external/gazetteer.json`), providing alias matching, campaign detection, and infrastructure grounding without penalizing out-of-vocabulary entities.

---

## Statistical NER Model Performance (CRF)

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

## Running Evaluation and Tests

```bash
# Run unit tests
python -m unittest discover -s tests

# Evaluate NER models on dev set
python notebooks/07_compare.py

# Rebuild BIO training datasets
python notebooks/05_build_bio.py

# Rebuild ATT&CK gazetteer
python notebooks/10_build_gazetteer.py
```
