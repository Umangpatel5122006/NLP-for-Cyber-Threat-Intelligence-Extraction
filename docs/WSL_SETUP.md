# Optional Windows Subsystem for Linux setup

WSL is an alternative when native Windows Python cannot import the classical
NLP stack or when you prefer Linux. It is not required on a working native
installation. These commands use your distribution and checkout path.

## Choose a distribution

In PowerShell, inspect installed distributions:

```powershell
wsl --list --verbose
```

Open your installed Ubuntu distribution from the Start menu or `wsl`. If WSL
is absent, use your organization's approved installation process. Distribution
startup errors happen before project Python runs.

Use Python 3.11 or 3.12 inside Linux. Ubuntu 24.04/Python 3.12 is validated;
the distribution's installed name can differ.

## Install inside Linux

In the **Linux terminal**, enter your checkout. Replace the example path:

```bash
cd "/mnt/c/path/to/Cyber_Threat_Detection"
python3 --version
python3 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install -r requirements-lock.txt
.venv-wsl/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv-wsl/bin/python -m pip check
.venv-wsl/bin/python scripts/start_workbench.py --train-models
```

On Ubuntu, missing `venv`/`ensurepip` requires the matching OS
`python3-venv` package through normal package management. You can also clone
inside the Linux filesystem and run the same commands there.

Use `.venv-wsl` for Linux and `.venv` for Windows if sharing a checkout.
Virtualenvs cannot be shared across operating systems. Both folders are ignored.
No activation or Windows execution-policy changes are needed.

Open **http://127.0.0.1:8765** in the Windows browser. Ctrl+C stops the app.
Subsequent Linux-terminal startup:

```bash
.venv-wsl/bin/python scripts/start_workbench.py
```

A checkout inside Linux can always be launched from its Linux terminal. Windows
launcher integration assumes WSL can reach the same checkout directory.

## Optional PowerShell launcher

From the Windows checkout root:

```powershell
python scripts/start_workbench.py
```

It tries native Python, then project `.venv-wsl/bin/python` in WSL's default
distribution. For another distribution, substitute its installed name:

```powershell
python scripts/start_workbench.py --wsl-distro "YourDistributionName"
```

For an environment elsewhere in Linux, pass an absolute Linux Python path
rather than a Windows path or a tilde:

```powershell
python scripts/start_workbench.py --wsl-distro "YourDistributionName" --wsl-python "/home/your-linux-user/envs/cti/bin/python" --check
```

These options configure WSL candidates; they do not override an already ready
native runtime. Inside Linux, `python -m cti.web` bypasses runtime search.

The launcher retains the previous fixed cache environment as a compatibility
fallback. New installations do not require it or a particular Linux username.

## Verify or prepare without starting a server

Inside the checkout's Linux terminal:

```bash
.venv-wsl/bin/python -m pip check
.venv-wsl/bin/python -m unittest discover -s tests -v
.venv-wsl/bin/python scripts/start_workbench.py --check
```

Direct model preparation:

```bash
.venv-wsl/bin/python -m cti.train_ner
.venv-wsl/bin/python -m cti.train_message --download
```

Train and run in the same environment. Manifests reject incompatible runtime
versions/resources; refit rather than bypassing those checks. NER preparation
does not require the optional relevance model: default hard gating is disabled.

Earlier machine-specific startup failures are historical records in
[preparation](../checkpoints/ACTION_BILOU_PREPARATION.md) and
[recovery](../checkpoints/ACTION_BILOU_EXPERIMENT.md), not required setup steps.
No Windows security settings are changed by this workflow. Use
[SETUP.md](SETUP.md) for package/runtime troubleshooting.
