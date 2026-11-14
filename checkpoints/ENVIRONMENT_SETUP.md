# Environment setup and baseline validation

Date: 2026-11-14 (Asia/Calcutta)

## Scope

This checkpoint follows recovery of the zero-filled working-tree files and Git
index. It sets up the core Python environment and makes the configured document
pipeline independent of unselected model libraries. Evaluation formulas, trained
models, dataset files, and previous experiment results are unchanged.

## Changes

- Created an ignored `.venv` using CPython 3.11.7 on Windows x64.
- Verified that all existing direct dependency pins are published on PyPI. Kept
  those pins and installed only the core dependencies.
- Added `requirements-lock.txt`, a snapshot of installed core, transitive, and
  build-tool versions. The editable local package is excluded from the snapshot.
  Other operating systems and Python versions have not been validated.
- Made package metadata read the existing core and optional requirements files.
  Python metadata now requires at least 3.11, matching the pinned pandas and
  scikit-learn requirements. Neural dependencies are available via `.[neural]`.
- Added lazy registration of the seven built-in document modules. Building a
  configured pipeline no longer scans and imports every NLP/model file.
  Explicitly registered custom modules remain usable. `load_all_modules()` is
  retained for callers intentionally importing all components.
- Replaced the print-only pipeline demo test with two assertions-based tests:
  extraction/offset checks and startup in a fresh interpreter without unselected
  NLP libraries.
- Moved optional neural test imports inside tests and added dependency-based
  skips. This allows the classical tests to be discovered without neural extras
  in an environment where the core native dependencies can load.
- Changed split-lock tests to temporary synthetic raw/BIO data and report IDs.
  Testing the authorization path no longer loads real held-out annotations.

## Validation actually performed

Raw command output is in `results/environment_validation.txt`.

| Check | Result |
|---|---|
| Core installation and editable package metadata | Passed |
| `python -m pip check` | Passed: no broken requirements |
| Dependency snapshot vs installed versions | Passed: all 64 pinned packages match |
| Python syntax and TOML parsing | Passed: 62 Python files and package configuration |
| Targeted pipeline suite | Passed: 2 tests |
| Full unittest discovery | Incomplete: 2 pipeline tests passed, 5 modules failed to import |

The failing imports originate from Windows Application Control blocking native
extensions, including spaCy's `srsly.ujson` and pandas' `nattype`. Later import
errors can also mention partially initialized pandas after its first failure.
The installed dependency graph is consistent; this does not establish that every
package can execute under the machine's application policy. No security settings
were changed. The model/preprocessing and evaluation suites remain unverified in
this environment, including the newly isolated optional tests and split fixtures.

## Review commands

Use the setup commands in the root README when recreating the environment. In
the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_pipeline.py -v
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
git diff --check
git diff --stat
```

If the full suite reports the same Application Control error, it still needs a
Python/package environment approved by the machine's policy before a complete
passing baseline can be recorded.

## Next step

Resolve the full-suite execution blocker, then correct evaluation in a separate
change: exclude `O` from token metrics, define one-to-one relaxed matching, and
add regression cases before regenerating affected metrics. Keep the official
test split reserved for final frozen evaluation.
