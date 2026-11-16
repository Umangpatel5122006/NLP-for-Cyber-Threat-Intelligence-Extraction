# Action annotation review pack

Date: 2026-11-16 (Asia/Calcutta)

## Completed step

The previously identified 21 conflicting-token groups now have a readable,
train-only casebook and a provenance-bound decision workbook. The source labels,
splits, default models and frontend/backend behavior are unchanged. No model
was fitted and no dev/test examples were evaluated. No new dataset was downloaded.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cti.action_review
```

The WSL equivalent also works. Native regeneration was actually checked and
preserved the existing decision workbook byte-for-byte. The tool needs the raw/
processed training files and the versioned reference summary, with no model
binaries required. When the frozen local OOF cache is absent, only the optional
model coverage counts are omitted. A cache that exists but has a wrong hash is
rejected. All output is restricted to a dedicated `results/` subdirectory.

## Actual findings

| Conflict type | Groups |
|---|---:|
| Action present versus absent | 11 |
| Same actions, differing extents | 6 |
| Span count / segmentation | 3 |
| Different non-overlapping actions | 1 |

The 21 groups contain 43 records: 32 relevant and 11 background. Nine groups
still contain conflicting annotations with identical preceding context. The
largest concentration is 28 rows in `95_mitre_report`; review the repeated
sections/code in context rather than treating rows as independent examples.

Only 57 of the 4,453 gold actions lie in these rows. Frozen baseline OOF has
2 exact matches there, with 557 exact matches / 4,396 gold actions elsewhere.
That means most exact-span errors remain outside the known conflicting groups;
the review is not proof that label correction alone will solve the model's
coverage. No masking or filtered F1 improvement is claimed.

Available token inputs force at least 21 exact whole-sentence label disagreements
for one fixed deterministic predictor; adding exact preceding context still
leaves nine. These are consistency lower bounds, not span-F1 ceilings or
estimates of errors caused by training. They do not apply directly to the
separately fitted cross-validation models.

## Concrete review files

- `results/action_review_v1/casebook.md`: all occurrences with source, context,
  readable gold spans, original offsets and focused questions. Same-context
  conflicts appear first. Dataset text stays inside escaped-length code fences.
- `results/action_review_v1/decisions.csv`: 21 **pending** group-level decisions.
  Reviewer/date/notes are required when a finding is recorded. Evidence/group
  fields are bound to the current source; stale, duplicate, missing or unknown
  decisions are rejected. Existing notes are never overwritten on regeneration.
- [The review guide](../docs/ACTION_ANNOTATION_REVIEW.md): examples and unresolved
  presence, extent, segmentation, code, nominal-action and context conventions.
- [Compact results](../results/action_review_v1/summary.json): counts, current
  review status, reference hashes and unchanged local artifact identities.

All source excerpts and editable decisions remain ignored by Git. The module,
tests, guide/checkpoint and compact summary are reviewable source artifacts.
No automatic label winner or corrected training data were produced. The earlier
43-row review file and its notes remain untouched; it currently has no notes.

## Validation and next action

All **341 WSL tests pass**, including 14 added synthetic tests for classification,
context consistency, train-only access before file loading, row/hash alignment,
workbook preservation, stale evidence, review metadata, output paths, safe source
rendering and optional-cache operation. Native pack generation also succeeds.
Model/configuration identities and raw/processed training hashes are checked
before/after generation. No new browser or remote CI run is claimed.

The agent has not performed the human decisions: all 21 remain pending. Start
with the first nine same-context groups, using the guide and source sections,
then record findings in the decision CSV and rerun the command to validate.
Any approved annotation convention/correction must be a separate versioned step
with a new training snapshot; the official benchmark labels are not rewritten.
The review does not establish detection accuracy or trained Hindi/Hinglish support.

Nothing was committed or pushed.
