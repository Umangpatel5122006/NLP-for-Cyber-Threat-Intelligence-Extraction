# Explicit action-end labels: prepared, runtime validation pending

Date: 2026-11-15 (Asia/Calcutta)

**Historical preparation record:** WSL startup was subsequently restored and
the experiment completed on 2026-11-16. See [the completed checkpoint](ACTION_BILOU_EXPERIMENT.md)
for the actual comparison, 327 passing WSL tests and baseline-retention decision.
The blocked/unrun statements below describe preparation time.

## Current outcome

The BILOU experiment, isolated artifact format, encoding audit and tests are
prepared. **The project training comparison has not run.** The prepared WSL
runtime now fails before launching Python, while native spaCy remains blocked
by Windows Application Control. There is no new CV/dev score or candidate
artifact, and no full-suite passing result for this step.

The existing default entity/action/SMS models, held candidates, pipeline/training
configuration, labels and splits are unchanged. The preceding ending-feature
candidate remains held. Current recorded dev exact F1 is still 0.2596 for
entities / 0.1551 for actions; BILOU has not improved those figures.

## Prepared classical experiment

The existing action CRF uses BIO labels, which do not explicitly mark the end
of a multi-token action. The new isolated `ActionBILOUCRFTagger` learns:

| Label | Meaning |
|---|---|
| B-ACTION | Beginning of a multi-token action |
| I-ACTION | Interior of a multi-token action |
| L-ACTION | Last token of a multi-token action |
| U-ACTION | A one-token action |
| O | Outside an action |

Input annotations and reported predictions remain BIO. The converter rejects
malformed gold labels and preserves every gold boundary exactly. Prediction uses
the CRF's unconstrained internal decode, then a fixed `open_span_v1` adapter:
B/U begin a span; I/L continue an open B/I, otherwise begin one; L/U close it.
An unfinished span closes at O/B/U or sentence end. This is a deterministic
normalizer, not a constrained Viterbi implementation or a tuned threshold.
The runner records how many internal sequences need normalization.

Features remain the current morphology/boundary-dictionary profile, with the
same sentence-excluded training dictionary statistics. The rejected ending-cue
features are not added. The only intended change is internal label encoding.
The implementation uses linear-chain CRF, fixed lexical rules and the existing
blank tokenizer. No neural model, embeddings, trained parser or LLM API is used.

`configs/action_bilou_experiment.yaml` fixes one candidate, three report-disjoint
training folds, seed 42, relevant-only fitting, c1/c2=1 and 200 iterations.
Reference OOF predictions must match the audited current-model snapshot.
Selection needs exact F1 gain >=0.005, wins in two folds, recall loss <=0.01 and
background predictions <=5% above baseline. The selection is written before any
possible dev access. Only a qualifying candidate receives full-train fitting
and dev confirmation (gain >=0.002 and the same recall/background budgets).
No automatic activation occurs.

Experimental BILOU weights use their own `classical_action_bilou_experiment`
artifact kind, with internal/output schemas, normalization version and resource/
source/runtime hashes. `load_candidate` is the explicit loader. The ordinary
NER loader rejects this format. Existing artifact checks are not weakened to
hide L/U states or load an unvalidated candidate into the app.

## Checks actually completed

The tokenizer-independent encoding audit passes on **all 8,635 train sentences /
70 reports / 223,042 tokens**. Every gold BIO sequence is identical after
BIO -> BILOU -> BIO. All 4,453 action spans are retained:

| Internal tag | Count |
|---|---:|
| O | 200,927 |
| B-ACTION | 3,469 |
| I-ACTION | 14,193 |
| L-ACTION | 3,469 |
| U-ACTION | 984 |

End plus unit counts equal 4,453. Source offsets and UID alignment pass the
existing train-data loader. Model/configuration hashes match the baseline;
no gold labels were rewritten and no dev/test examples were evaluated.

**Fourteen new native tests pass:** eight conversion/normalization tests and
six real-CRF plumbing tests. The latter explicitly substitute tiny toy feature
matrices to test internal states, BIO output, empty positions, artifact round
trips, corruption detection and source-drift rejection without the tokenizer.
They do not test the production feature matrix, current trained models or
project detection accuracy. The production code has no toy-feature fallback.

The existing **82 native API/language/Indic-rule/combined-findings/intake tests**
also pass. Independent post-run auditor helpers agree with the production
normalizer/validity check over all **3,125 five-token internal label paths**.
All seven new Python files parse successfully. These results do not replace a
production-stack test or an actually completed post-run model audit.

## Blocked checks and exact evidence

- Both the experiment launch and a bare VM probe,
  `wsl -d Ubuntu-24.04 --exec /bin/true`, fail with
  `Wsl/Service/CreateInstance/CreateVm/HCS/0xc0000139`. Ubuntu-24.04 is installed
  as WSL2 and stopped. The failure occurs before the project's Python starts.
  The underlying Windows/WSL cause has not been diagnosed here.
- Native import fails in `srsly.msgpack._packer` with
  `An Application Control policy has blocked this file.`
- The production test module therefore fails at import. Its three test cases
  remain pending; they are not counted as passing. Training-fold fits, dev
  confirmation, production persistence and the full post-run audit are unrun.
- The latest full-suite pass is the **previous** 310-test WSL run in
  [the ending-feature checkpoint](ACTION_ENDING_EXPERIMENT.md). The expanded
  current full suite has not passed or been rerun in a compatible environment.

No WSL reset/update, Windows restart or security-policy change was performed.
Logs remain ignored: `results/action_bilou_wsl_check.txt`,
`action_bilou_native_tests.txt`, `action_bilou_native_regressions.txt` and
`action_bilou_production_tests.txt`. Compact evidence is in
[the preparation report](../results/action_bilou_v1/preparation/summary.json)
and [encoding-only integrity audit](../results/action_bilou_v1/preparation/integrity_audit.json).

## Resume and review

First verify that Ubuntu can launch from the normal PowerShell terminal:

```powershell
wsl -d Ubuntu-24.04 --exec /bin/true
```

Once that succeeds, from the project root run:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.experiment_action_bilou'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python scripts/audit_action_bilou.py'
```

The final command requires actual completed root-level CV/selection/summary
files. The preparation audit is deliberately separate and cannot stand in for
it. A successful encoding audit is not authorization to claim a model gain.

Native checks that already work:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_action_bilou.py -v
.\.venv\Scripts\python.exe -X utf8 scripts/audit_action_bilou_encoding.py
```

The training runner needs the compatible, audited local background-experiment
snapshot and ignored OOF caches. These are absent from a fresh clone;
`--train-models` alone does not recreate historical experiment prerequisites.
Do not bypass runtime/resource manifests or retrain the active models just to
make the comparison load. No dataset download is needed for the prepared run.

Review the isolated model/format, pure converter, frozen protocol, runner,
audit scripts and tests. Full validation and training remain required before
any activation. Nothing was committed or pushed.
