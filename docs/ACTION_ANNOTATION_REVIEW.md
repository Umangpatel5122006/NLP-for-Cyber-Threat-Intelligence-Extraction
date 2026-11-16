# Reviewing the action labels

This review concerns English **ACTION span extraction from security reports**.
SMS/scam classification and Hindi/Hinglish message rules are separate outputs.
An ACTION annotation says where a described behavior occurs; it does not by
itself prove that a message is malicious or that a reported claim is true.

The original labels, official splits and model artifacts stay fixed. This
workflow records findings about conflicting annotations; it does not choose a
winning annotation, export training corrections or retrain anything.

## Start with the generated evidence

From the project root, either prepared runtime can build the pack:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cti.action_review
```

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.action_review'
```

Open these local files:

- `results/action_review_v1/casebook.md`: all 21 groups / 43 occurrences, ordered
  with same-context conflicts first. It shows the source text, preceding context,
  report/UID, relevance and readable original action spans with character offsets.
- `results/action_review_v1/decisions.csv`: one editable record per group. Edit
  only `decision`, `reviewer`, `reviewed_on` and `notes`.

The source excerpts, including shell commands, are evidence to read. Generation
does not execute their contents. Keep notes in the decision CSV; the casebook is
regenerated. Re-running preserves an existing decision CSV byte-for-byte and
validates it against the original evidence. Changed or missing evidence is
rejected rather than silently replacing reviewed records.

Only raw/processed **train** data and the versioned reference report are required.
Local model binaries are not needed. If the ignored frozen OOF cache is available,
the tool adds hash-checked aggregate coverage counts. A teammate without that
cache can still review the same evidence; coverage is reported unavailable.

## Questions to settle consistently

These are open policy questions for review, not approved corrections to the
benchmark or instructions to automatically remove data.

| Question | Evidence in the current conflicts | Decision needed |
|---|---|---|
| Action present or absent? | A repeated screenshot caption labels `Shared` once and nothing elsewhere. Repeated page-navigation text also differs. | Decide whether the occurrence describes a target behavior, document metadata, or an extraction artifact. |
| Where does an action end? | Identical context accompanies `to cover up malicious or unauthorized SSH access` versus the same span including `to the device`. | Specify inclusion of objects, targets, prepositional arguments, modifiers and auxiliary/purpose words. |
| One action or several? | An exploitation sentence is one broad span in one record and two shorter action spans in another. | Decide how independent predicates, coordinated clauses and purpose clauses are segmented. |
| How is code annotated? | Repeated shell-code text labels command names in one occurrence and commands with arguments in another. | Define whether the target is a command name, a complete operation, a capability, or prose only. Check the source report if flattened extraction obscures command boundaries. |
| Are nominal behaviors included? | Existing gold examples include phrases such as `initial access`, without a leading verb. | Record how nominal behaviors are treated; a verb-only rule is not equivalent to the current corpus. |
| Does context explain a difference? | Some repeated sentences have different preceding sections; nine groups still conflict with identical preceding context. | Distinguish missing broader context from inconsistent annotations. Do not assume preceding-context features resolve every group. |

Use the same decisions across related groups. Existing model predictions are not
the authority for which annotation is correct. The casebook contains original
annotations and context; predictions are used only for aggregate impact counts.

## Record the review

Leave the group/evidence fields unchanged. Set one decision:

| Value | Meaning |
|---|---|
| `pending` | No review recorded yet. |
| `context_explains` | Explain the contextual distinction and where it is visible. |
| `inconsistent_labels` | Describe the conflicting label/extent convention and the evidence. |
| `source_extraction_issue` | Describe suspected caption, boilerplate, duplicated section or flattened-code problems; distinguish suspicion from verified source evidence. |
| `uncertain` | Record the unresolved question or source material still needed. |

For a non-pending decision, fill `reviewer`, `reviewed_on` as `YYYY-MM-DD`, and
explanatory `notes`. A review finding is not an approved corrected gold sequence.
The tool deliberately has no apply/relabel/training operation.

Example of a review note:

> Both occurrences have the same sentence and preceding context. One includes
> the device target and one excludes it. Our span-extent convention is unresolved;
> consult the source section before proposing a corrected training copy.

That note supports `uncertain`; it does not claim either label is correct. No
actual decisions have been filled automatically. Once notes are entered, run
the generation command again to validate and refresh the compact decision counts.

Any later correction needs an explicit convention, its own data snapshot and
review provenance. Official test labels remain reserved. A changed annotation
policy cannot be compared to earlier F1 figures as if the target were unchanged.

## What this review can establish

There are 11 presence/absence groups, six boundary-extent groups, three span-
count/segmentation groups, and one group selecting different actions. Nine
groups still conflict with identical preceding context. That is evidence of
input/label inconsistency at the available context level, not proof that every
difference is an annotation mistake.

The 43 rows contain **57 of 4,453 gold actions (1.28%)**. Frozen baseline OOF
predictions match two of those actions exactly; 557 exact matches and 4,396 gold
actions are outside the conflict rows. Most exact-span errors therefore remain
outside these known groups. These counts do not bound how training noise can
affect learned weights elsewhere, and they do not establish a causal diagnosis.

For a single fixed deterministic token-only predictor, the observed variants
force at least 21 whole-sentence tag disagreements across these rows. Even with
identical preceding context supplied, at least nine remain. This is **not a
span-F1 ceiling**, an accuracy estimate or a bound on the report-fold models,
which are separately fitted. No rows were removed from scoring or training.

The pack makes this review concrete. Broader missed-action and boundary coverage
still need work after the target convention is clear; cleaning these 43 records
alone is not evidence that the current extractor will become reliable.
