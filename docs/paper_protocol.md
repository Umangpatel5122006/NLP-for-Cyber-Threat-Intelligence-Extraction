# Evaluation Protocol and Benchmark Comparison

Updated 2026-11-16. New evaluation summaries and exports use **metrics version 2**.
Historical token and relaxed-span figures must be recomputed before comparison
with version 2. The original result files are preserved as experiment history.

## 1. Project Evaluation Protocol

This project uses a multi-perspective evaluation protocol implemented in
`src/cti/eval/`, with explicit split and metric scopes to reduce leakage and
misleading comparisons.

### A. Headline vs. Diagnostic Scopes
1. **Headline Scope (All Supplied Evaluation Sentences)**:
   - Evaluates predictions across **all sentences** in the evaluation corpus, including irrelevant sentences (`relevant: "no"`).
   - In production CTI feeds, reports arrive without human relevance labels. A deployable system must either classify relevance accurately or suppress false positives on background text.
   - This covers false positives on background sentences, but the inputs are
     dataset sentences with processed token-aligned BIO gold. It does not
     evaluate the full raw-report sentence splitter, original character-span
     annotation convention, or the combined message/IOC/review workbench.
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
   - Per-label counts use the same parser as the selected mode. Orphaned `I-`
     tags can therefore score differently in default and strict IOB2 mode.
2. **Relaxed-Span Matching (`relaxed`)**:
   - A predicted entity is considered a True Positive if it **overlaps** with a ground-truth entity of the same class ($\text{start}_{\text{pred}} < \text{end}_{\text{gold}}$ and $\text{start}_{\text{gold}} < \text{end}_{\text{pred}}$).
   - Reflects operational utility in cybersecurity triage (e.g., recognizing "Cobalt Strike Beacon" when gold was "Cobalt Strike" still alerts analysts to the correct malware).
   - Matching is one-to-one within each sentence and label, maximizing the number
     of overlapping pairs. One predicted span cannot recall two gold spans, and
     two fragments cannot both receive credit for a single gold span. Precision
     and recall use the same true-positive pair count. Half-open spans that only
     touch at a boundary do not overlap.
3. **Token-Level Metrics (`token_level`)**:
   - Token-level micro-F1 and macro-F1 across entity categories (`THREAT_ACTOR`, `MALWARE_TOOL`, `INFRASTRUCTURE`, `ACTION`), excluding the negative class (`O`).
   - Labels are the union of observed gold and predicted entity types for the
     evaluated subset, excluding `O`. Background tokens remain in the aligned
     sequences: entities predicted on `O` count as false positives, and gold
     entities predicted as `O` count as false negatives. BIO boundary prefixes
     are removed only for this token-level diagnostic.
   - Empty or all-`O` inputs return zero scores with the normal result schema.
4. **Structured Error Taxonomy**:
   - Every entity is categorized into one of five mutually exclusive states:
     - `Exact Match`: identical boundaries and label.
     - `Boundary Error`: overlapping span with matching label but differing boundaries.
     - `Wrong Type`: overlapping span with conflicting entity labels (type confusion).
     - `Missed`: unmatched ground-truth entity after exact, boundary, and type-error pairing.
     - `Spurious`: unmatched prediction after that pairing.

All metric functions reject unequal sentence counts and unequal token counts
within a sentence. Equal flattened token totals do not establish alignment.

---

### C. Split Isolation and Statistical Rigor
1. **Test Split Lock**:
   - The official 20-report test split (`data/splits/test_report_ids.json`) is strictly locked behind `CTI_ALLOW_TEST=1`.
   - All iterative development, tuning, and validation occur exclusively on the 80 train + dev reports.
2. **Report-Grouped K-Fold Cross-Validation**:
   - Current action experiments group strictly on `doc_id` across the 70 train
     reports, then confirm only a selected candidate on the 10 dev reports.
     Earlier M3 tuning across 80 train + dev reports is historical development
     evidence, not an untouched validation estimate.
   - Sentences from the same report never appear in both train and validation folds.
   - The current helper uses seeded report shuffling and greedy sentence-count
     balancing across folds; it does not call scikit-learn's `GroupKFold`.
3. **Cluster Bootstrapping**:
   - 95% empirical confidence intervals are computed by resampling whole reports (clusters) with replacement (1,000 iterations), accounting for document-level correlation.

---

## 2. Comparison with the SMAH-CTI Paper

The [official released paper](https://github.com/davidmohaisen/SMAH-CTI/blob/main/emnlp-smah-cti.pdf)
was checked on 2026-11-14. Sections 4.8, 5.1, and 5.3 specify all-sentence NER
evaluation and exact entity-type/character-offset matching. Table 4 reports
protected-test NER F1 of 0.2074 (RoBERTa), 0.1414 (SecBERT), 0.1753
(multi-task RoBERTa), and 0.2233 (GPT-4o-mini). These are reference results;
this project is not adding the paper's LLM approach.

The earlier 74–81 NER assumption is not supported by that table. This resolves
the premise of the supposed 40-point gap, without establishing that our models
outperform the paper.

Direct comparison still requires matching the split, all four entity types,
and the original character-span convention. This repository currently scores
separate entity/action layers on cleaned, token-aligned BIO gold; punctuation
trimming, token expansion, and overlap filtering can change the evaluated spans.
Those scores are not the paper's original-character-span benchmark.

Paper SHA-256 at verification:
`5209cde2ba775dcdb318b5c1231351704927a3b4270613246dfa1373b4a57407`.

## 3. Historical results and regeneration

Existing default exact-span results on valid, aligned BIO sequences are not
changed by these fixes. Historical token scores included `O`, and historical
relaxed scores allowed multiple overlapping spans to claim credit. Recompute
those diagnostics from saved predictions or a reproducible model rerun; their
corrected values cannot be recovered from aggregate CSV scores alone.

No benchmark tables were regenerated for this change. Synthetic regression
cases and export tests validate the corrections without training models or
evaluating the protected test split. New reports carry `metrics_version: "2"`
in the summary, a version column in CSV, and a version line in Markdown.
