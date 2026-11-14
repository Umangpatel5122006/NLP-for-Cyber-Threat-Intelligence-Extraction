# Checkpoint: Phase 5 (Milestones M0, M1, M2 Complete)

**Date**: 2026-11-14  
**Status**: Milestones M0, M1, and M2 completed and verified.

---

## 1. Summary of Completed Milestones

### Milestone 0: Baseline Lock & Repository Hygiene
- **Dependencies**: Pinned core packages (`requirements.txt`) and optional deep learning libraries (`requirements-optional.txt`).
- **Git Hygiene**: Configured `.gitignore` (caches, `/models/`, temporary results; kept all raw/external datasets tracked). Untracked `.pyc` files from index.
- **Baseline Lock**: Captured exact baseline outputs across 11 unit tests and models in `results/baseline_before_changes.txt`.
- **Learning Note**: Authored `docs/learning/M0_baseline_hygiene.md`.

### Milestone 1: Leakage and Data Audit
- **Gazetteer Audit (K6)**: Audited `notebooks/10_build_gazetteer.py` and eliminated 13 hand-added terms leaking into dev/test (`scilc.exe`, `telegram`, `active directory`, `plc`, `domain controller`, `github`, etc.).
- **Campaign Separation**: Moved 57 ATT&CK campaigns out of `THREAT_ACTOR` into a distinct `CAMPAIGN` group.
- **Provenance Tracking**: Added `data/external/gazetteer_provenance.json` tracking source attribution for 1,813 terms.
- **Leak Impact Check**: Documented in `results/gazetteer_leak_check.md` that removing `scilc.exe` dropped CRF `MALWARE_TOOL` F1 by 0.013, while MEMM improved by +0.008 F1.
- **Test Split Lock**: Updated `load_smah` and `load_bio` to require `CTI_ALLOW_TEST=1`. Added unit tests.
- **Unlabelled Mention Audit**: Detected 32,270 unlabelled mentions in irrelevant sentences; exported a 100-mention sample to `results/unlabelled_mentions_sample.csv`.
- **Learning Note**: Authored `docs/learning/M1_leakage_and_data_audit.md`.

### Milestone 2: Evaluation Harness (K5)
- **Multi-Perspective Metrics (`src/cti/eval/metrics.py`)**: Exact seqeval (default and strict IOB2), relaxed span overlap, token-level macro/micro F1, and a 5-class error taxonomy (exact match, boundary error, wrong type, missed, spurious).
- **Report-Grouped Cross-Validation (`src/cti/eval/cv.py`)**: `GroupKFold` over the 80 train+dev reports, with explicit test split exclusion guard.
- **Cluster Bootstrapping (`src/cti/eval/bootstrap.py`)**: Resamples entire reports for empirical 95% confidence intervals.
- **Reporting Utilities (`src/cti/eval/report.py`)**: Slices metrics across 5 document sources (`aptnotes`, `crowdstrike`, `mandiant`, `mitre`, `sentinel`) and exports to Markdown and CSV.
- **Script Refactoring**: Updated `notebooks/07_compare.py` and `notebooks/12_relevant_only.py`.
- **Documentation**: Authored `docs/paper_protocol.md` and `docs/learning/M2_evaluation_harness.md`.
- **Test Coverage**: 26 unit tests now passing in `tests/`.

---

## 2. Benchmark Metrics Record (Dev Set, Clean Gazetteer)

| Model | Scope | Metric Mode | Precision | Recall | F1 Score |
|---|---|---|---:|---:|---:|
| **CRF (`crf_tagger`)** | Headline (All Dev) | Exact-Span (default seqeval) | 0.236 | 0.282 | **0.257** |
| **CRF (`crf_tagger`)** | Headline (All Dev) | Exact-Span (strict IOB2) | 0.236 | 0.282 | **0.257** |
| **CRF (`crf_tagger`)** | Headline (All Dev) | Relaxed-Span (Overlap) | 0.309 | 0.381 | **0.341** |
| **CRF (`crf_tagger`)** | Headline (All Dev) | Token-Level Macro | 0.425 | 0.516 | **0.458** |
| **CRF (`crf_tagger`)** | Diagnostic (Relevant Dev) | Exact-Span (default seqeval) | 0.421 | 0.282 | **0.338** |
| **CRF (`crf_tagger`)** | Diagnostic (Relevant Dev) | Relaxed-Span (Overlap) | 0.551 | 0.381 | **0.451** |
| **CRF (`crf_tagger`)** | Diagnostic (Relevant Dev) | Token-Level Macro | 0.590 | 0.521 | **0.551** |
| **MEMM (`memm_tagger`)** | Headline (All Dev) | Exact-Span (default seqeval) | 0.224 | 0.316 | **0.262** |
| **MEMM (`memm_tagger`)** | Diagnostic (Relevant Dev) | Exact-Span (default seqeval) | 0.419 | 0.316 | **0.360** |

---

## 3. Next Step: Milestone 3 (CRF Tuning)
1. Feature extraction caching via `joblib`.
2. Stage 1: Coarse random search on regularization (`c1`, `c2`), `feature.minfreq`, and training algorithms (`lbfgs`, `l2sgd`, `ap`, `pa`, `arow`).
3. Stage 2: Top configs evaluated with 5-fold report-grouped CV.
4. Stage 3: Feature-group ablation.
5. Stage 4: New feature exploration (document consistency, two-pass predictions, word clusters).
