# Evaluation correctness fixes

Date: 2026-11-14 (Asia/Calcutta)

## Scope and changes

This step corrects scoring and report consistency before further NER tuning.
It does not change training data, features, model weights, or old result files.

- Token metrics remove `O` from the union of gold/predicted label types. They
  retain entity-vs-background errors rather than deleting background tokens.
- Relaxed scoring uses maximum one-to-one overlap pairing within each label and
  sentence. Its precision and recall share one true-positive count. Ordered,
  disjoint BIO spans allow a linear two-pointer matcher.
- Strict IOB2 per-label counts now use seqeval's strict parser, matching the
  overall strict score. Default exact scoring is unchanged for valid BIO data.
- All metrics reject sentence-count and per-sentence token-count mismatches.
- Empty inputs and all-background subsets return zero metrics with consistent
  fields, including empty relevant-only subsets during report export.
- Report generation checks metadata/mask lengths. New summary, CSV, and Markdown
  reports identify metrics version 2. Unversioned input summaries are labeled
  `legacy/unspecified` on export instead of being silently tagged as version 2.
- The paper protocol now records the verified released paper's character-span
  scoring and actual NER baseline table, replacing the unsupported 74–81 claim.

## Concrete regression examples

These examples use synthetic BIO tags, not benchmark results.

| Example | Previous behavior | Corrected behavior |
|---|---|---|
| Miss the only entity among three background tokens | Token micro-F1 0.75 | Token micro-F1 0 |
| One broad prediction overlaps two gold entities | Relaxed F1 1 | P=1, R=0.5, F1=0.6667 |
| Two fragments overlap one gold entity | Relaxed F1 1 | P=0.5, R=1, F1=0.6667 |
| Predict orphaned `I-ACTOR` against `B-ACTOR` | Strict overall F1 0 but per-label F1 1 | Both strict scores 0 |

## Validation

- Added 16 regression tests to the 10 existing evaluation tests.
- All **26 evaluation/report/CV/bootstrap tests passed** in the core `.venv`.
- The matching test checks all 1,156 pairs of valid four-token, single-label BIO
  sequences against an independent exhaustive pairing search.
- CSV/Markdown export, per-source diagnostics, zero-entity subsets, and metadata
  validation are exercised using temporary synthetic reports.
- Full discovery: **28 tests passed; 3 test modules failed to import** because
  Windows Application Control still blocks spaCy's `srsly.ujson` extension.
  Those modules are `test_bio`, `test_models_preprocessing`, and `test_tokenizer`.
  No security settings were changed and no unavailable tests were marked passing.
- Raw full-suite output is saved in `results/evaluation_validation.txt`.

## Review commands

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_eval_metrics.py -v
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
git diff --check
git diff --stat
```

## Results policy and next step

Token and relaxed-span values in existing experiment tables are historical and
need regeneration under version 2. Their aggregate scores cannot be corrected
without the original predictions or a rerun. No new model performance is claimed.

The next checkpoint is to validate spaCy/tokenization in an approved runtime,
record the full baseline, and integrate saved entity/action taggers into the
document pipeline. Regenerated development reports should use a new versioned
output prefix so the old results remain available for comparison. Keep the
official test split reserved for final frozen evaluation.
