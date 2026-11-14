# Evaluation Protocol and Benchmark Comparison

## 1. Project Evaluation Protocol

To ensure rigorous, reproducible, and leakage-free evaluation of Cyber Threat Intelligence (CTI) entity extraction, this project establishes a standardized multi-perspective evaluation protocol implemented in `src/cti/eval/`.

### A. Headline vs. Diagnostic Scopes
1. **Headline Scope (All Sentences End-to-End)**:
   - Evaluates predictions across **all sentences** in the evaluation corpus, including irrelevant sentences (`relevant: "no"`).
   - In production CTI feeds, reports arrive without human relevance labels. A deployable system must either classify relevance accurately or suppress false positives on background text.
   - All headline metrics represent true end-to-end extraction effectiveness.
2. **Diagnostic Scope (Relevant Sentences Only)**:
   - Evaluates predictions strictly on ground-truth relevant sentences (`relevant: "yes"`).
   - Serves as an isolated diagnostic tool to evaluate sequence tagger capabilities (CRF, MEMM, etc.) independent of upstream sentence filtering errors.
   - **Must always be explicitly labeled as a diagnostic** and never conflated with headline performance.

---

### B. Evaluation Dimensions and Metrics
1. **Exact-Span Matching (`exact_default` and `exact_strict`)**:
   - Implemented via `seqeval`.
   - A predicted entity is a True Positive if and only if both the **token boundaries** (`[start, end)`) and the **entity label** match the ground-truth span exactly.
   - Reported in both standard `seqeval` mode and strict IOB2 mode (`mode='strict', scheme=IOB2`).
2. **Relaxed-Span Matching (`relaxed`)**:
   - A predicted entity is considered a True Positive if it **overlaps** with a ground-truth entity of the same class ($\text{start}_{\text{pred}} < \text{end}_{\text{gold}}$ and $\text{start}_{\text{gold}} < \text{end}_{\text{pred}}$).
   - Reflects operational utility in cybersecurity triage (e.g., recognizing "Cobalt Strike Beacon" when gold was "Cobalt Strike" still alerts analysts to the correct malware).
3. **Token-Level Metrics (`token_level`)**:
   - Token-level micro-F1 and macro-F1 across entity categories (`THREAT_ACTOR`, `MALWARE_TOOL`, `INFRASTRUCTURE`, `ACTION`), excluding the negative class (`O`).
4. **Structured Error Taxonomy**:
   - Every entity is categorized into one of five mutually exclusive states:
     - `Exact Match`: identical boundaries and label.
     - `Boundary Error`: overlapping span with matching label but differing boundaries.
     - `Wrong Type`: overlapping span with conflicting entity labels (type confusion).
     - `Missed`: ground-truth entity with zero overlapping predictions (False Negative).
     - `Spurious`: predicted entity with zero overlapping ground-truth annotations (False Positive).

---

### C. Split Isolation and Statistical Rigor
1. **Test Split Lock**:
   - The official 20-report test split (`data/splits/test_report_ids.json`) is strictly locked behind `CTI_ALLOW_TEST=1`.
   - All iterative development, tuning, and validation occur exclusively on the 80 train + dev reports.
2. **Report-Grouped Cross-Validation (`GroupKFold`)**:
   - K-fold cross-validation is grouped strictly on `doc_id` across the 80 train + dev reports.
   - Sentences from the same report never appear in both train and validation folds.
3. **Cluster Bootstrapping**:
   - 95% empirical confidence intervals are computed by resampling whole reports (clusters) with replacement (1,000 iterations), accounting for document-level correlation.

---

## 2. Comparison with the SMAH-CTI Paper

- **Reported Paper Numbers**: The SMAH-CTI paper baseline NER scores were informally referenced as roughly 74–81 F1.
- **Observed Repository Scores**:
  - Exact-span seqeval on the official 10-report dev set yields ~0.26 F1 on all sentences, and ~0.34 F1 on relevant sentences.
  - Relaxed-span F1 on the dev set reaches ~0.42–0.45.
  - Token-level macro F1 reaches ~0.45–0.50.
- **Verification Statement**:
  - The exact scoring code and evaluation protocol used in the paper (e.g., whether scoring was token-level, relaxed-overlap, or exact-span, and whether irrelevant sentences were filtered prior to scoring) cannot be verified from the codebase.
  - Per project instructions, we do not guess or fabricate baseline protocols. Our benchmark numbers are recorded from real, reproducible code executions under the protocol defined above.
