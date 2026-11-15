# Original-report evidence offsets

Date: 2026-11-15 (Asia/Calcutta)

## Result

The document pipeline now preserves the supplied report as `original_text` and
projects normalized sentence/entity offsets back to it. Cleaning and defanging
produce the same normalized text as before. No model training, feature change,
new dataset, or official test evaluation was needed for this step.

`results/offset_demo.json` is the updated output for
`examples/sample_report.txt`, using the existing saved entity/action CRFs.
For example, the IOC's normalized text is `185.22.10.5`, while its
`original_text` is `185.22.10[.]5`.

## Output contract

| Field | Meaning |
|---|---|
| Document `text` | Current cleaned/defanged text used by NLP modules |
| Document `original_text` | Exact supplied Unicode string before preprocessing |
| Entity/sentence `text`, `start`, `end` | Evidence and offsets in Document `text` |
| Entity/sentence `original_text`, `original_start`, `original_end` | Source evidence and offsets in Document `original_text` |
| `metadata.offsets` | Schema version 1, Unicode code-point units, half-open intervals and covering-span policy |

For every emitted entity and sentence:

```python
document.text[start:end] == text
document.original_text[original_start:original_end] == original_text
```

The JSON schema is extended with fields; normalized positions retain their
meaning. Existing NER metadata still says `offset_space: document_text`.
Confidence remains unchanged, including `null` for NER.

Positions count **Unicode code points**, with the end position excluded. They
are not UTF-8 byte offsets or JavaScript UTF-16 indices. A future browser client
can slice `Array.from(document.original_text)` with these positions, or convert
them explicitly before using browser range APIs. Emoji tests cover this detail.

Original spans are **covering intervals**, not a lossless inverse of arbitrary
substring normalization. Each changed regex replacement inherits the complete
matched source interval. Unicode composition/reordering can also require a
covering interval that includes an intervening combining mark. Source intervals
may overlap. This is deliberate evidence provenance; it does not claim a
one-to-one character correspondence or that normalizing every projected source
substring will reproduce the selected normalized substring exactly.

## Implementation

- `core/offsets.py` holds character provenance through NFC, regex substitutions,
  deletions, whitespace collapse and trimming. NFC uses standard-library Unicode
  normalization with canonical decomposition alignment; it handles combining
  mark reordering, expansion and Hangul composition.
- `Document.original_span(start, end)` projects a nonempty normalized range.
  Internal per-character mappings stay out of dataclass/JSON serialization.
- `Document.transform_text(...)` is the supported path for text-changing
  modules. It preserves the original report and rejects changes after sentence
  or entity extraction, so existing annotations cannot silently become stale.
  Direct untracked assignment to text is rejected when the pipeline validates it.
- Cleaner and defanger use tracked transformations. Their normalization order
  and regex rules are unchanged. `metadata.text_before_defang` retains its
  earlier meaning: the text immediately before the first defang step, which
  may already have been cleaned.
- IOC and NER modules attach original fields to findings, including standalone
  module use and repeated NER inference. Pipeline completion validates all
  evidence before attaching original fields to entities and sentences.
  Custom modules that build entities directly receive projection at pipeline
  completion; their text and offsets must agree.
- The CLI decodes UTF-8 file bytes without newline translation, preserving a BOM
  and CRLF in `original_text`. Its stdout now explicitly uses UTF-8, fixing a
  Windows encoding failure exposed by an emoji report. This changes no Windows
  security or system encoding setting.

## Validation completed

- **96 tests pass** in the prepared WSL Python 3.12 environment, including 18
  new offset tests and the strengthened two-layer NER integration test. Full
  output: `results/offset_validation.txt`.
- The same **18 offset tests pass on Windows Python 3.11** without importing
  the blocked spaCy stack: `results/offset_windows_validation.txt`.
- Tests cover repeated identical indicators at different source positions,
  defang rules, whitespace, deleted invisibles, NFC composition/reordering,
  Hangul, Indic joiners, emoji/code-point positions, empty reports, CRLF/BOM
  file input, stale annotations, untracked mutations and idempotent reruns.
- Normalized text agrees with the previous normalization operations for all
  **8,635 train + 1,238 dev sentences**. All 1,120 individual Unicode code points
  changed by NFC in the WSL runtime retain valid provenance. Another 3,000 seeded
  mixed-character samples match standard NFD ordering and previous normalized
  output. See `results/offset_integrity_validation.txt`.
- The actual saved-CRF demo retains the earlier normalized text, findings,
  positions and confidence. Every new original entity/sentence span matches
  the source report. Model binaries remain unchanged.
- No test gold data was selected or evaluated. No dependency pins or data files
  changed. The complete native Windows NLP stack remains subject to its earlier
  Application Control restriction; WSL remains the working full-suite runtime.

The previous committed NER step (`a77f5a8`) now has a successful
[GitHub Actions run](https://github.com/Umangpatel5122006/NLP-for-Cyber-Threat-Intelligence-Extraction/actions/runs/43639592315):
78 tests, relevance persistence, both real CRF training/reload runs, and document
inference passed on Ubuntu/Python 3.11. Its logs reproduce all-dev exact F1
0.2596 (entities) and 0.1379 (action).

This step adds original-evidence assertions to the CI inference check. The
updated workflow was checked locally; its remote run still awaits commit/push.

## Review commands

From the project root in PowerShell:

```powershell
# Offset tests run in the existing Windows environment.
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_offsets.py -v

# Complete classical suite and real saved-model demo use the prepared WSL environment.
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.analyze --input examples/sample_report.txt --output results/offset_demo.json'
Get-Content results/offset_demo.json
git diff --check
git status --short
```

No retraining or dataset download is needed to review this step. Use
[WSL_SETUP.md](../docs/WSL_SETUP.md) if recreating the environment.

Suggested commit: `feat: preserve original report evidence offsets`.
Commits and pushes remain with the user.

Next reviewable step: define IOC/NER merging while retaining source evidence
and legitimate action/entity overlaps. Frontend highlighting can then build on
the original-span contract. Accuracy work, technique classification and the
final frozen test evaluation remain separate tasks.
