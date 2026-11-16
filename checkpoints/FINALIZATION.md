# Finalization and presentation preparation

Date: 2026-11-16 (Asia/Calcutta)

## Changes

The README is now a current project overview with concise Windows/Linux setup,
first-run model preparation, one-command startup, capabilities, dataset roles,
separate measured results and explicit language/domain limits. Historical
experiments remain in checkpoints instead of dominating installation guidance.

`docs/SETUP.md` covers prerequisites, native environments, direct training,
offline presentation, CLI modes, browser QA and troubleshooting. WSL instructions
use a project `.venv-wsl` and the user's installed distribution; no particular
Linux username or cached interpreter path is required. `docs/README.md` indexes
the public guides. `docs/MODELS_AND_RESULTS.md` distinguishes active CRFs/SMS SVM,
optional relevance, rules and held candidates. The evaluation protocol no longer
calls sentence/BIO evaluation a complete original-report end-to-end benchmark;
it also distinguishes historical train+dev CV from current train-only selection.

The launcher now accepts `--wsl-distro` / `--wsl-python`, retains the older cache
as a compatibility fallback, and prefers a runtime with both requested NER/IOC
mode and SMS ready. Missing SMS is explicit if no fully ready runtime exists.
Paths are positional shell arguments, including spaces/metacharacters; user
values are not inserted into shell source. `.venv-wsl/` is ignored.

An abnormal readiness-process exit cannot be accepted just because JSON was
printed first. Training failures report their process exit code and do not
start the server. Temporary fitted/reloaded CRF readers are explicitly closed
between training layers and at function completion. A regression verifies that
cleanup is idempotent and predictions reopen without changing their output.

## Private learning folder

Root `learning/` contains an index and nine study chapters: problem/contribution,
dataset sources/labels/partitions, classical concepts/model choices, pipeline/code,
metrics/calculations, challenges/limitations, 40 evaluator questions, demo plan,
and commands/glossary. The language is suitable for presentation preparation and
identifies both successes and limits. It is not application input.

`/learning/` is ignored and no file there is tracked. This is distinct from
existing public `docs/learning/` historical research notes. The private material
will not be present in a normal fresh clone; public setup docs are self-contained.

## Actual validation

1. Created a separate Linux Python 3.12 environment and installed every pinned
   dependency from `requirements-lock.txt`; `pip check` passed.
2. Copied current nonignored source/data into an isolated checkout without
   ignored models, downloads or private learning notes. Installed it editable.
   Initial checks correctly reported NER/SMS absent; the no-model rules CLI worked.
3. **368 tests passed**, with successful process exit, in that environment before
   production model preparation. Six added launcher tests and one resource-lifecycle
   test cover the actual changes. All 16 launcher tests also pass natively.
4. The real `--train-models` launcher trained/reloaded both CRFs and the SMS SVM
   from missing artifacts, then started both UI/API modes. All recorded CTI dev
   exact metrics and the SMS dev confusion matrix reproduced. A second separate
   NER training process completed with exit 0 and reproduced both layers again.
5. Six real HTTP cases exercised English/Hindi/Hinglish positive rules in both
   modes. **33 real Chrome check groups passed** against fresh artifacts,
   including source offsets, downloads, both scopes/modes, disagreement, invalid
   and stale states, 26 Indic authored cases, keyboard operation and mobile layout.
6. Default/custom-WSL launcher readiness, JavaScript syntax, documentation links,
   Git whitespace/ignore rules and original artifact identities were checked.
   Detailed evidence is in [the compact report](../results/finalization_v1/README.md).

Training in this check used train/dev only. It did not refit or replace the
original active model files. No labels, official splits or default parameters
changed, and protected CTI/SMS test evaluation was not enabled. This is actual
local validation; no remote CI success is asserted. No security policy changed.

## Native-process finding and evidence limit

Initial clean preparation encountered intermittent native process crashes: one
during combined training and another after single-layer results were written
(signal 11). Other pre-change repetitions succeeded. The precise native fault
was not isolated, so it is not attributed to a particular upstream package or
Windows/WSL repair. Deterministic reader cleanup and explicit failed-exit handling
were added. Both subsequent NER training processes, model reloads and browser
checks completed successfully. These observations do not prove that no native
runtime fault can ever recur. Original failure logs remain ignored for diagnosis.

## Review and scientific limits

Follow [the public setup guide](../docs/SETUP.md), run `--check`, then start the
app and inspect each built-in example, task scope, source excerpt and JSON export.
Read the local learning index before the presentation; do not force-add it.

This finalizes the current research prototype's setup/documentation and checked
behavior. It does not create production-grade or flawless detection. English is
the only trained NLP language; Hindi/Hinglish remain phrase rules. CTI dev exact
F1 stays 0.2596/0.1551 and SMS dev spam F1 stays 0.9810. The 21 annotation-review
decisions and independent real multilingual data remain research work. Protected
test partitions stay reserved for a separately frozen final evaluation. Commits
and pushes remain with the user.
