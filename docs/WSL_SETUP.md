# Working classical NLP environment in WSL

Revalidated on 2026-11-16 using the existing Ubuntu-24.04 distribution, Python
3.12.3 and `requirements-lock.txt`. The environment is outside the repository at
`~/.cache/cti-classical-py312`. The Windows Python 3.11 environment still encounters
Application Control import failures; no Windows security setting was changed.

**Latest runtime check:** after the user reported recovery, Ubuntu starts and
the prepared Python 3.12.3 / spaCy 3.8.16 runtime imports successfully. All
**327 tests pass in WSL**, including the three BILOU production-stack checks
that were previously blocked. The native Windows import still encounters
Application Control, now reported in `srsly.ujson`; use the working WSL runtime.
The agent did not change Windows security settings. Earlier failure records
below are historical, not the current startup state.

### Earlier startup failure and follow-up on 2026-11-16

The earlier `HCS/0xc0000139` failure was confirmed in the user's normal
PowerShell terminal before recovery. Local
checks find WSL 3.0.5.0, the current stable release checked against Microsoft's
[release page](https://github.com/microsoft/WSL/releases/tag/3.0.5), and running
WSL/compute/network services. `wsl --shutdown` succeeds; `wsl --update` reports
that the most recent version is already installed. A subsequent bare Ubuntu
launch still fails with the same HCS error. Neither standard reboot-pending
registry marker is set; this does not rule out a problem resolved by restarting.

Microsoft identifies `0xc0000139` as
[`STATUS_ENTRYPOINT_NOT_FOUND`](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/596a1078-e883-4972-9bbc-49e60bebca55).
This establishes the error type, not which DLL or component is responsible.
The failure is before the project's Python starts. Native Python's separate
Application Control block was also confirmed in Code Integrity events.

The suggested next troubleshooting step was Microsoft's
[Windows component repair sequence](https://support.microsoft.com/en-us/windows/experience/backup-recovery/using-system-file-checker-in-windows).
Open **PowerShell as administrator**, run DISM first, and wait for successful
completion before running SFC:

```powershell
DISM.exe /Online /Cleanup-Image /RestoreHealth
sfc /scannow
```

Save work, restart Windows manually, then test:

```powershell
wsl -d Ubuntu-24.04 --exec /bin/true
```

If troubleshooting a recurrence and a repair command fails, preserve its final
output for diagnosis before continuing. WSL startup is now restored; the exact
repair actions/output were not provided, so no specific cause or fix is asserted.
These commands target Windows components; retain the existing Ubuntu
installation and project environment. The agent session is not elevated, so
DISM/SFC were not run. No restart, distribution reset or security-policy change
was performed. Local restart/update attempt logs are ignored at
`results/wsl_recovery_20261116/attempts.json`. See the current checkpoint for
the resumed model comparison; the old attempt log records the earlier failure.

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
needs this training because model binaries are not tracked by Git. It also
prepares the missing SMS classifier from the pinned official UCI corpus.
`--train-message-model` prepares only the SMS model before startup.
`--ioc-only` starts IOC plus message rules and the available classical SMS model,
preferring a compatible message-model runtime without loading NER.
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

To train the classical SMS model directly in this environment:

```powershell
wsl -d Ubuntu-24.04 --exec bash -c '~/.cache/cti-classical-py312/bin/python -m cti.train_message --download'
```

Only training CV and dev are evaluated. See [data provenance](SMS_DATASET.md)
and [the integrated checkpoint](../checkpoints/INTEGRATED_FINDINGS_AND_SMS_MODEL.md).

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
