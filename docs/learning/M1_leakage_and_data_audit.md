# Learning Note M1: Data Leakage, Gazetteer Provenance, and the Unlabelled-Mention Problem

## 1. The Big Picture
In applied Natural Language Processing (NLP) for Cyber Threat Intelligence (CTI), machine learning models (like CRFs and MEMMs) do not operate in a vacuum. They rely heavily on **domain knowledge sources**, such as external dictionaries (gazetteers of threat actors, malware tools, and infrastructure) and **dataset annotations**.

However, two major data integrity risks frequently corrupt research outcomes and inflate benchmark scores:
1. **Gazetteer Leakage (Data Contamination)**: If an external dictionary contains terms that were added by looking at the evaluation or test sets, the model appears to perform well during testing simply because it memorized answers in advance.
2. **Missing Label Noise (The Unlabelled-Mention Problem)**: In datasets like SMAH-CTI, human annotators only labeled entities in sentences they deemed "relevant" (`relevant: "yes"`). Irrelevant sentences were left completely unannotated (`O` for all tokens), even when they contain valid mentions of threat actors, malware, or tools. When a model correctly identifies an entity in an irrelevant sentence, standard evaluation penalizes it as a **false positive**.

Milestone 1 audits and permanently fixes both issues before any model tuning begins.

---

## 2. What We Changed and Why

### A. Gazetteer Audit and Provenance (`notebooks/10_build_gazetteer.py`)
- **What Changed**:
  - Audited all hand-added phrases in `10_build_gazetteer.py` and removed 13 leaked terms that appeared directly in dev/test gold annotations (`scilc.exe`, `telegram`, `active directory`, `plc`, `domain controller`, `github`, `svchost.exe`, `c2 server`, `c&c server`, `c2`, `c&c`, and generic actor phrases).
  - Separated 57 active ATT&CK campaigns (`type == "campaign"`, 65 alias forms) out of `THREAT_ACTOR` into a distinct `CAMPAIGN` group.
  - Implemented length filtering (`len >= 3`) and English stop-word filtering (`ENGLISH_STOP_WORDS`).
  - Added strict split-isolation enforcement (`--train-only`) so any term mined from labelled data comes exclusively from the training split.
  - Exported `data/external/gazetteer_provenance.json` mapping all 1,813 gazetteer terms to their exact origin (`ATT&CK:intrusion-set`, `ATT&CK:malware`, `ATT&CK:tool`, `ATT&CK:campaign`, or `curated:infrastructure`).
- **Why**: Eliminates artificial score inflation and guarantees that the test split remains completely uncontaminated.

### B. Test Split Lock (`src/cti/core/dataset.py`, `src/cti/models/data.py`)
- **What Changed**:
  - Modified `load_bio` and `load_smah` to raise a `PermissionError` if split `"test"` is accessed without the environment variable `CTI_ALLOW_TEST=1`.
  - Added unit tests in `tests/test_models_preprocessing.py` verifying that unauthorized test access is rejected and authorized access succeeds.
- **Why**: Prevents accidental snooping or evaluation on the test split during intermediate development.

### C. Unlabelled-Mention Audit (`notebooks/13_audit_unlabelled.py`)
- **What Changed**:
  - Scanned 6,433 irrelevant sentences across train and dev splits for mentions of known training gold entities.
  - Identified **32,270 unlabelled entity mentions** (13,414 Action, 8,280 Infrastructure, 6,411 MalwareTool, 4,165 ThreatActor).
  - Exported a reproducible 100-mention sample to `results/unlabelled_mentions_sample.csv` for qualitative review.
- **Why**: Confirms empirically that irrelevant sentences contain thousands of genuine entities that were never annotated, explaining why all-dev precision is lower than relevant-dev precision.

---

## 3. Alternative Techniques (Classical vs Modern)

| Dimension | Classical Approach (What We Did) | Modern / Production Alternative |
|---|---|---|
| **Gazetteer Grounding** | Exact string and phrase matching using SpaCy `PhraseMatcher` and rule-based regex. | **Entity Linking / Wikification**: Using bi-encoders (e.g. BLINK) or dense vector retrieval (FAISS) over MITRE ATT&CK embeddings to match names and synonyms even with typographical errors or descriptive paraphrases. |
| **Data Leakage Prevention** | Static provenance JSON tracking and explicit environment variable locks (`CTI_ALLOW_TEST=1`). | **Automated Data Governance**: Feature stores (e.g. Feast) with point-in-time correctness, and immutable dataset artifact hashing (e.g. DVC / MLflow data lineage). |
| **Unlabelled Entity Handling** | Pipeline filtering: train a binary relevance classifier first to discard irrelevant sentences before passing text to the NER tagger. | **Positive-Unlabeled (PU) Learning / Confident Learning**: Algorithms (e.g. Cleanlab) that statistically re-weight loss functions for unlabelled tokens, treating them as unlabelled rather than definitive negatives (`O`). |

---

## 4. How to Swap in an Alternative (One-Line Config)
In the CTI pipeline configuration (`configs/config.yaml`), the gazetteer category behavior can be adjusted or toggled:

```yaml
# In configs/config.yaml or model parameters:
gazetteer:
  include_campaigns: false   # Swap to true to re-enable campaign names in the feature extractor
  min_length: 3
```

To run gazetteer generation with train entities included without touching dev or test:
```bash
python notebooks/10_build_gazetteer.py --train-only --include-train-entities
```
