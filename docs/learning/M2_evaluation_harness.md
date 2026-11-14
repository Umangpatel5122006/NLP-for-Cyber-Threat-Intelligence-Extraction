# Learning Note M2: Multi-Tier Evaluation Harness, Error Taxonomies, and Cluster Bootstrapping

## 1. The Big Picture
In Natural Language Processing (NLP), evaluating sequence tagging models (such as Named Entity Recognition for Cyber Threat Intelligence) with a single metric—like standard exact-match F1 on a single split—creates significant blind spots:
1. **The Exact-Match Penalty**: If an analyst's ground truth labels `"Cobalt Strike"` as `MALWARE_TOOL` and the model predicts `"Cobalt Strike Beacon"` as `MALWARE_TOOL`, exact-span metrics count this as both a **False Negative** (missed the exact gold span) and a **False Positive** (predicted an ungrounded span). Yet in operational cybersecurity triage, detecting the tool correctly is highly valuable.
2. **Document Correlation (The Cluster Effect)**: Sentences from the same report share the same writing style, vocabulary, and entities (e.g. Lazarus, APT29). Evaluating or bootstrapping by randomly splitting sentences produces overly optimistic scores and unrealistically narrow confidence intervals.
3. **Headline vs. Diagnostic Scopes**: In real CTI feeds, reports are uncurated. An evaluation that only measures relevant sentences overstates real-world performance because it ignores false positives on irrelevant sentences.

Milestone 2 builds an evaluation harness (`src/cti/eval/`) that addresses all three challenges.

---

## 2. What We Changed and Why

1. **Multi-Perspective Metrics (`src/cti/eval/metrics.py`)**:
   - Implemented exact-span matching (`seqeval` default and strict IOB2).
   - Implemented **relaxed-span matching** (token overlap with matching label), isolating boundary precision from entity detection.
   - Implemented **token-level micro/macro F1**, enabling direct comparison with token-level literature.
2. **Structured Error Taxonomy (`classify_errors`)**:
   - Rather than just reporting a single F1 number, predictions are categorized into:
     - `Exact Match`: Perfect span and label.
     - `Boundary Error`: Overlapping span with the correct entity label.
     - `Wrong Type`: Overlapping span with conflicting entity labels (e.g., confusing malware with infrastructure).
     - `Missed`: Ground-truth entity missed entirely.
     - `Spurious`: Prediction hallucinated on background text.
3. **Report-Grouped Cross-Validation (`src/cti/eval/cv.py`)**:
   - Created `get_report_grouped_splits` using `GroupKFold` over the 80 train+dev reports.
   - Guaranteed that test split reports (`test_report_ids.json`) can never be included.
4. **Cluster Bootstrapping (`src/cti/eval/bootstrap.py`)**:
   - Computes 95% confidence intervals by resampling whole reports (clusters) with replacement, properly estimating variance across independent documents.
5. **Per-Source Breakdown and Reporting (`src/cti/eval/report.py`)**:
   - Slices metrics across the 5 corpus sources (`aptnotes`, `crowdstrike`, `mandiant`, `mitre`, `sentinel`).
   - Exports structured reports in both Markdown and CSV to `results/`.
6. **Refactored Exploration Scripts**:
   - Updated `notebooks/07_compare.py` and `notebooks/12_relevant_only.py` to use the unified harness.

---

## 3. Alternative Techniques (Classical vs Modern)

| Dimension | Classical Approach (What We Did) | Modern / Production Alternative |
|---|---|---|
| **Span Evaluation** | Exact token boundary matching + relaxed overlap matching via rules. | **MUC-style & SemEval Fuzzy Metrics**: Partial credit matrices (e.g., SemEval 2013 Task 9) giving fractional scores (0.5) for boundary shifts vs type mismatches. |
| **Resampling** | Non-parametric cluster bootstrap by resampling document clusters with replacement. | **Bayesian Hierarchical Modeling**: Fitting a mixed-effects logistic regression model with document-level random effects to estimate posterior parameter distributions. |
| **Error Diagnostics** | Rule-based error categorization into missed, spurious, boundary, and type confusion. | **Automated Behavioral Testing (CheckList)**: Behavioral tests evaluating model robustness under perturbation (invariance, directional expectation, typo injection). |

---

## 4. How to Swap in an Alternative (One-Line Config)
To toggle between evaluation perspectives when running model comparisons or tuning:

```powershell
# Run exact, relaxed, and token-level evaluation with CSV/Markdown report export
python notebooks/12_relevant_only.py --layer entities --models crf_tagger --export-prefix eval_run1
```

To run cross-validation with report-grouping in code:
```python
from cti.eval.cv import get_report_grouped_splits

# Generate 5-fold cross-validation partitions grouped strictly by report:
folds = get_report_grouped_splits(doc_ids, n_splits=5, seed=42)
```
