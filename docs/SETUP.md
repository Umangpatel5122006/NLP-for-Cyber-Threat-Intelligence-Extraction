# Installation and operating guide

Run commands from the repository root: the directory containing `pyproject.toml`,
`configs/` and `data/`. Paths below contain no contributor-specific usernames.

## Requirements

- Python 3.11 or 3.12, Git and a modern browser. A GPU is unnecessary.
- Disk space for the environment, included CTI corpus and local model files.
- Internet for packages and the initial pinned UCI SMS download. Later text
  analysis does not need internet.
- Node and Chrome/Chromium are optional, for browser automation only.

The dependency snapshot originated on Windows Python 3.11 and has also been
installed and exercised under Ubuntu Python 3.12. macOS and newer Python versions
are not validated platforms here. Install from a source checkout: configs,
corpus, examples and dictionaries are not packaged in a standalone wheel.

Clone your repository URL, then enter its directory. Replace both placeholders:

```bash
git clone <repository-url>
cd <repository-directory>
```

## Windows PowerShell

No activation or execution-policy change is required:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/start_workbench.py --train-models
```

Use `py -3.11` if that is the supported version installed on your machine. If
`py` is unavailable, use your installed Python's full path to create the venv.

If compiled NLP imports are blocked, use [WSL](WSL_SETUP.md) or another permitted
Linux environment. The project does not require disabling security controls.

## Linux / macOS

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python scripts/start_workbench.py --train-models
```

Check the version before creating the environment. On Ubuntu, if venv creation
reports missing `ensurepip`, install the matching `python3-venv` OS package
through normal package management, then retry.

If package installation cannot build `seqeval`, install the pinned build tools
first and retry the lock file. Match `requirements-lock.txt` rather than loosening
pins. Below, `python` means the environment's executable:

```bash
python -m pip install setuptools==84.2.1 wheel==0.48.0
python -m pip install -r requirements-lock.txt
```

For unsupported CPU/OS wheels, prefer the validated Linux setup. An unavailable
wheel is distinct from a model prediction error.

## First startup and model preparation

`--train-models` first looks for compatible saved models. If necessary, it
trains entity/action CRFs using `configs/ner_training.yaml` on CTI train,
scores dev and saves artifacts under `models/`. It also prepares an unavailable
English SMS classifier: download the hash-pinned corpus, compare three classical
profiles using train-only grouped CV, freeze selection, fit train and score dev.
It does not evaluate protected test partitions. Compatible models are reused.
Initial training may take several minutes depending on hardware.

Tracked data/configuration required by a fresh checkout:

- `data/raw/SMAH-CTI.json`
- `data/processed/bio_en_{entities,action}_{train,dev}.json`
- `data/splits/` report IDs and frozen SMS manifests
- `data/external/gazetteer.json` and feature-resource files
- `configs/ner_training.yaml`, `config_ner.yaml` and `config.yaml`

Protected CTI data are included for eventual frozen evaluation; their presence
does not authorize routine test evaluation. Restore missing tracked files from
the same repository revision rather than substituting unreviewed data.

For preparation before a demo, direct training commands return when complete
instead of starting a server:

```bash
python -m cti.train_ner
python -m cti.train_message --download
python scripts/start_workbench.py --check
python scripts/start_workbench.py
```

Here and below, `python` means `.venv/bin/python` on Linux/macOS or
`.\.venv\Scripts\python.exe` on Windows unless the environment is activated.
Run from the checkout root. Open **http://127.0.0.1:8765**; Ctrl+C stops the app.
Subsequent startup needs only `python scripts/start_workbench.py`.

## Modes and command-line use

```bash
# Check only; never trains or starts, even with training flags.
python scripts/start_workbench.py --check
# Change port; open the matching localhost URL.
python scripts/start_workbench.py --port 8766
# No trained NER required. Rules and IOCs still run.
python scripts/start_workbench.py --ioc-only
# Explicitly prepare an unavailable SMS classifier.
python scripts/start_workbench.py --ioc-only --train-message-model
# Full configured report pipeline; requires NER artifacts.
python -m cti.analyze --input examples/merge_report.txt --output results/demo.json
# Basic configured rules + IOCs; no saved models required.
python -m cti.analyze --config configs/config.yaml --input examples/message_risk_demo.txt --output results/rules-demo.json
# Serve directly using this interpreter, bypassing runtime search.
python -m cti.web --port 8765
```

The basic CLI config excludes the optional SMS classifier; browser IOC mode
additionally attempts to load it. Both include message rules. The browser has
an explicit language selector; CLI analysis currently uses Auto.

The launcher tries project `.venv`, current Python and Windows WSL candidates.
It prefers a runtime with the requested extraction mode plus SMS ready;
otherwise SMS unavailability is explicit. To force an interpreter, use
`python -m cti.web` directly. WSL options are explained in [WSL_SETUP.md](WSL_SETUP.md).

Models check Python minor version, package versions and relevant feature
resources. Train and run in the same environment. `--train-models` prepares
missing/incompatible artifacts; it does not refit a compatible model solely
because training configuration was edited. For an intentional configured refit,
run `python -m cti.train_ner`. Keep trusted artifacts and manifests together.

## Offline presentation

Prepare dependencies and models while online. Confirm `--check` reports both
modes and `message_classifier.ready: true`, then start before presenting.
There is no remote inference or API key. Missing SMS models produce explicit
partial coverage while rules remain available. Inputs over 1,000 characters
are outside the SMS classifier's scope; report extraction can still run.
Hindi/Hinglish rules do not enable English-trained CRFs/SVMs in those languages.

## Verification

```bash
python -m pip check
python -m unittest discover -s tests -v
```

Unit tests fit small synthetic models as needed; downloaded production models
are unnecessary. Test-lock authorization tests use synthetic data. Fresh-checkout
training/inference is a separate integration check.

Optional JavaScript checks:

```bash
node --check src/cti/web/static/app.js
node --check scripts/check_web_ui.mjs
```

With the server running, specify your installed Chrome/Chromium executable:

```powershell
node scripts/check_web_ui.mjs 'C:\Program Files\Google\Chrome\Application\chrome.exe' http://127.0.0.1:8765 results/browser-review
```

```bash
node scripts/check_web_ui.mjs /usr/bin/chromium http://127.0.0.1:8765 results/browser-review
```

Browser QA requires compatible saved NER/SMS artifacts. Logs, downloads and
screenshots stay ignored. The [CI workflow](../.github/workflows/classical-nlp.yml)
also trains on train/dev; a workflow file does not establish a successful run.

## Troubleshooting

| Symptom | Action |
|---|---|
| `No module named cti` | Install `-e .` with the Python that starts the app. |
| NER setup required | Use `--train-models` in a compatible installed environment. |
| SMS missing/incompatible | Use `--train-message-model` or direct `cti.train_message --download`. |
| Runtime/resources changed | Retrain there; retain manifest checks. |
| Compiled import blocked | Use permitted WSL/Linux without changing Windows security policy. |
| Port occupied | Stop your old process or use `--port 8766`. |
| Old-response notice | Stop the old server, restart and reload the browser. |
| Hindi/Hinglish model coverage limited | Expected: phrase rules exist; learned models do not. |
| No positive detection | Inspect coverage; absence does not establish safety. |
| Download failed | Retry with internet; verify pinned provenance before any manual replacement. |
