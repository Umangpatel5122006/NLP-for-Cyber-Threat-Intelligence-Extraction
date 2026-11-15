# Working classical NLP environment in WSL

Validated on 2026-11-15 using the existing Ubuntu-24.04 distribution, Python
3.12.3 and `requirements-lock.txt`. The environment is outside the repository at
`~/.cache/cti-classical-py312`. The Windows Python 3.11 environment still encounters
Application Control import failures; no Windows security setting was changed.

## Use the prepared environment

Run these from the project root in PowerShell. WSL inherits the repository's
working directory; the non-login shell expands the Linux home path.

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m pip check'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m unittest discover -s tests -v'
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.analyze --input examples/sample_report.txt --output results/ner_demo.json'
```

For the browser frontend and backend together:

```powershell
python scripts/start_workbench.py
```

The launcher checks the project virtualenv and current Python before the
prepared WSL interpreter, and requires saved NER readiness by default.
`--check` only reports setup. `--train-models` explicitly trains missing or
incompatible artifacts in a working runtime before starting. A teammate's clone
needs this training because model binaries are not tracked by Git.
`--ioc-only` starts IOC plus English scam-pattern analysis without NER.
Direct startup in this prepared environment is still supported:
`wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.web'`.

Open http://127.0.0.1:8765 in Windows; Ctrl+C stops the server. See
[the browser checkpoint](../checkpoints/BROWSER_WORKBENCH.md) for UI review steps.

The saved NER artifacts were trained in this environment. Loading deliberately
checks Python minor version, relevant package versions, and feature resources.
Retrain after changing those dependencies; do not bypass the manifest checks.
If models are absent after a fresh clone, run training first:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.train_ner'
```

NER training does not require the separate relevance artifact because hard
sentence gating is disabled in `configs/config_ner.yaml`.

## Recreate the environment

Inside Ubuntu, change to the repository directory under `/mnt/c/`, then run:

```bash
python3 --version
python3 -m venv --without-pip "$HOME/.cache/cti-classical-py312"
python3 -m pip --python "$HOME/.cache/cti-classical-py312/bin/python" install -r requirements-lock.txt
"$HOME/.cache/cti-classical-py312/bin/python" -m pip install --no-build-isolation --no-deps -e .
"$HOME/.cache/cti-classical-py312/bin/python" -m pip check
```

The recreation commands assume the same installed Ubuntu Python 3.12 and system
pip used during validation. They install project packages into the isolated
environment, without changing Windows policy or globally installing project
dependencies. Do not use a login shell when relying on WSL to inherit the
project directory.
