# Consolidated IOC/NER findings

Date: 2026-11-15 (Asia/Calcutta)

## Result

The configured pipelines now finish with `result_merger`, producing a
deterministic findings view while retaining the detector observations in
`document.entities`. This is classical rule-based postprocessing. No model
training, new dependencies, dataset changes or official test evaluation were
needed.

The existing saved-CRF sample has five observations consolidated into four
findings: the exact-span IP and INFRASTRUCTURE observations share one finding
with both sources retained. See `results/merge_demo.json`.

The additional synthetic report `examples/merge_report.txt` demonstrates
mixed-case defanged URL/email hosts and repeated IP mentions. Its actual
saved-model output, `results/merge_report_demo.json`, has **13 observations,
9 findings, 7 primary findings and 2 host components**. Every distinct source
observation remains available as evidence.

## Merge policy

| Situation | Policy |
|---|---|
| Same span, label, source and confidence repeated | One evidence record in the view; raw observations retained |
| Same span and label from different sources/scores | One finding with separate evidence records |
| One supported specific IOC and generic INFRASTRUCTURE at the exact same span | Specific IOC label, with both original labels and evidence retained |
| Multiple conflicting specific IOC types at that span | Separate findings and conflict edges; no generic-label absorption |
| Larger INFRASTRUCTURE span surrounding an IOC | Keep both; record the overlap |
| Domain or IP exactly matching a URL's parsed host | Retain as a linked component of the URL |
| Domain exactly matching an email's domain | Retain as a linked component of the email |
| Domain-like text in a URL path/query or email local part | Keep as primary; containment alone does not imply a host component |
| Action/entity overlap | Keep both as primary; record `action_overlap` |
| Other conflicting or crossing spans | Keep findings with `label_conflict`, `contained_overlap` or `crossing_overlap` |
| Same canonical value at different positions | Separate occurrences with separate evidence |

These edges describe overlap structure, not semantic attack relations. Host
components are still present in `findings`; a frontend may show primary results
first and expand components. Broader entity ambiguity remains visible for review.

## Output contract

- `entities`: detector observations, including existing normalized/original
  positions, source labels and confidence.
- `findings`: consolidated occurrences with `id`, `text`, `label`, normalized
  and original positions/text, `canonical_value`, `canonical_status`, `sources`,
  `observed_labels`, and copied `evidence`. Each has a `role` of `primary` or
  `component`, plus `parent_ids` / `component_ids`.
- `overlaps`: `left_id`, `right_id`, and `kind`, referencing finding IDs.
- `metadata.merge`: schema version 1, policy `conservative_occurrence_v1`,
  observation/evidence/finding/component counts and confidence policy.

Finding IDs are local to one result and deterministic for the same observations,
regardless of input order. They are not persistent database identifiers; adding
an earlier occurrence can renumber later findings.

No aggregate confidence is created. Evidence keeps the original per-source
scores, including NER's `null`; rule-based IOC `1.0` is not interpreted as a
probability that an attack occurred. Internal evidence copies do not mutate the
raw observations. Source spans retain the [offset contract](OFFSET_MAPPING.md):
Unicode code points, half-open intervals and covering original spans.

## Canonical values

Canonical values are additional comparison/display keys; they never replace
evidence text or its positions. They normalize IP representation, ASCII domain
case/root dots, hash case and CVE case. URL scheme/host case is normalized while
credentials, explicit port spelling, path, query, fragment and empty delimiters
are retained. Email local-part case is preserved and its domain normalized.

This is syntax normalization, not an existence or maliciousness check. Supported
values have `canonical_status: available`. Unsupported/malformed IOC syntax,
including Unicode domain names without an explicit IDNA policy, remains visible
with `unavailable`. Non-IOC entity types use `not_applicable`. Absolute authority
URLs are supported; scoped IPs and other URL forms retain raw evidence without
an invented canonical value. No network lookup or URL visit occurs.

## Integration details

`result_merger` is lazily registered under `cti.output.merger`. It runs last in
the IOC-only, relevance and saved-NER configs. Pipeline construction rejects a
misplaced or repeated merger. Custom pipeline additions must be inserted before
the final merger.

NER/IOC reruns invalidate the derived view, so callers must rerun the merger
after changing detector output. Pipeline execution does this in the configured
order. Input spans, source types and confidence are validated before replacing
an existing merged view; invalid input cannot publish a partially rebuilt view.
Custom code that directly changes observations must call
`document.invalidate_findings()` and rebuild the view.

Integration testing also exposed an earlier IOC alignment defect: `ioc-finder`
normalizes hostname/email case, but exact case-sensitive lookup lost mixed-case
source mentions. Matching now handles those normalized cases while retaining
the actual text and preserving case-sensitive URL credentials/path/query/fragment.
Separate `/A` and `/a` URL mentions are covered by a regression test.

## Validation completed

- **121 tests passed in WSL**, including 25 new merger tests and strengthened
  NER rerun/invalidation coverage. See `results/merge_validation.txt`.
- **25 merger tests passed on Windows Python 3.11**, using the existing
  environment without changing security settings. See
  `results/merge_windows_validation.txt`.
- Both real saved-model demos retain every distinct observation in finding
  evidence. All finding/evidence positions match normalized/original text;
  component references are reciprocal and counts agree. The earlier sample's
  raw entities, confidence and text remain identical. Both CRF binaries are
  unchanged. See `results/merge_integrity_validation.txt`.
- No dependency pins, data files or earlier result artifacts changed. The
  official test split remains reserved. The complete native Windows NLP stack
  still has the earlier Application Control restriction; WSL runs the full suite.

The preceding offset commit (`3bf5cae`) has a successful
[GitHub Actions run](https://github.com/Umangpatel5122006/NLP-for-Cyber-Threat-Intelligence-Extraction/actions/runs/43679041828):
96 tests, real relevance/CRF persistence and original-offset inference checks
passed. This step adds finding/evidence/reference assertions to CI. Its updated
workflow is locally checked; the remote run awaits commit/push.

## Review commands

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_merger.py -v
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.analyze --input examples/merge_report.txt --output results/merge_report_demo.json'
Get-Content results/merge_report_demo.json
git diff --check
git status --short
```

The saved models are already present locally. No dataset download or retraining
is needed for this review. See [WSL_SETUP.md](../docs/WSL_SETUP.md) for a fresh setup.

Suggested commit: `feat: consolidate IOC and NER findings with source evidence`.
Commits and pushes remain with the user.

## Frontend readiness

After review of this step, a basic frontend can be the next step. The backend
now provides saved-model inference, original-report highlighting positions,
consolidated results and expandable source evidence/components. A first UI can
accept a report, show primary findings, highlight original evidence and let the
user inspect overlaps and detector sources. It should use the existing Python
pipeline rather than duplicate detection rules in the browser.

This is readiness for a prediction-review demo, not completion of the research
project. The current all-dev exact F1 remains 0.2596 for entities and 0.1379 for
actions. Action boundary improvements, technique classification, system/
vulnerability extraction, semantic relations and final frozen evaluation remain
backend/model work. A review UI can help inspect those errors while that work
continues; no score improvement is claimed by this merge step.
