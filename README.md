# CTI Workbench: classical NLP for messages and threat reports

A local application for reviewing suspicious messages and extracting evidence
from cyber threat intelligence (CTI) reports. Paste text or upload a UTF-8
`.txt`/`.md` file, inspect source highlights and Findings, and export JSON.

The frontend is HTML/CSS/JavaScript served by a Python backend. **One command
starts both.** No npm, API key or pretrained language-model download is required.
All active analysis uses **classical NLP and machine learning**.

## What it does

| Task | Methods | Output |
|---|---|---|
| Message assessment | Word/character TF-IDF + linear SVM; contextual phrase rules | SMS ham/spam prediction, scam-pattern findings and source cues |
| Report extraction | Two CRFs; IOC patterns | Actors, malware/tools, infrastructure, actions and technical indicators |
| Evidence review | Source-offset tracking and conservative merging | Original excerpts, detector sources, overlaps, components and JSON |

Message assessment and report extraction have separate cards and filters.
Finding an IP address does not establish maliciousness. Zero findings does not
establish safety. The app supports analyst review; it does not inspect network
traffic, verify senders, fact-check claims or block attacks.

| Capability | English | Hindi / Hinglish | Other languages |
|---|---|---|---|
| Trained SMS classifier | Yes, SMS up to 1,000 characters | Unavailable | Unavailable |
| Trained entity/action CRFs | Yes | Unavailable | Unavailable |
| Message phrase rules | Yes | Yes | Unavailable |
| Technical IOC patterns | Available across surrounding languages | Available | Available |

Auto language selection is an unverified script/lexical heuristic. Use manual
English/Hindi/Hinglish selection if it guesses incorrectly.

## Quick start

Use Python **3.11 or 3.12** and Git. Run commands from the checkout root.
The full runtime was validated on Linux/WSL with Python 3.12.3.
See [the setup guide](docs/SETUP.md) for installation details and troubleshooting.

**Windows PowerShell** (activation is unnecessary):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/start_workbench.py --train-models
```

Use `py -3.11` if that is your installed supported version. If native compiled
imports are blocked, use the [optional WSL guide](docs/WSL_SETUP.md) without
changing Windows security settings.

**Linux / macOS terminal**:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python scripts/start_workbench.py --train-models
```

The macOS commands use the same Python tooling but have not been validated on
a Mac in this project. See the setup guide for unavailable dependency wheels.

First startup with `--train-models` prepares missing/incompatible NER and SMS
artifacts on **train/dev only** and downloads the pinned UCI SMS corpus.
Training may take several minutes. Model binaries are ignored by Git. Internet
is needed for package installation and the first SMS download; analysis runs
locally afterward. The CTI corpus, BIO files and split manifests are included.

Open **http://127.0.0.1:8765**. Try **Message example** and **Report example**,
select **Analyze report**, inspect both task cards and filter Findings.
Press Ctrl+C in the terminal to stop.

For subsequent starts, use the installed environment's Python:

```bash
python scripts/start_workbench.py
```

Here and in the guides, `python` means `.venv/bin/python` on Linux/macOS or
`.\.venv\Scripts\python.exe` on Windows unless that environment is activated.

## Useful commands

```bash
# Check readiness; does not train or start a server.
python scripts/start_workbench.py --check
# Start on another port.
python scripts/start_workbench.py --port 8766
# IOCs + message rules; use the SMS classifier if available.
python scripts/start_workbench.py --ioc-only
# Prepare only an unavailable SMS model.
python scripts/start_workbench.py --train-message-model
# Full report pipeline; requires NER artifacts.
python -m cti.analyze --input examples/merge_report.txt --output results/demo.json
# Full suite; downloaded production models are unnecessary.
python -m unittest discover -s tests -v
```

See [setup](docs/SETUP.md) for direct training, a rules/IOC CLI and browser QA.

## Datasets and measured results

| Dataset / resource | Use |
|---|---|
| [SMAH-CTI](https://github.com/davidmohaisen/SMAH-CTI) | English relevance and NER; official report-level splits |
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection) | Active SMS model; template-grouped partitions |
| MITRE ATT&CK snapshot / gazetteer | Dictionary features, not reputation verification |
| [Mendeley SMS supplement](docs/SMS_DOMAIN_DATA.md) | Expanded-data candidate; held and not active |
| [Audited Indic sources](docs/HINDI_HINGLISH_DOWNLOADS.md) | Development diagnostics; no trained Hindi/Hinglish model |

**Recorded development results**, not final-test or universal threat accuracy:

| Active task | Evaluation scope | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Entity CRF | All 1,238 CTI dev sentences; exact token-aligned spans | 0.2470 | 0.2735 | 0.2596 |
| Action CRF | Same dev sentences; exact token-aligned spans | 0.1419 | 0.1711 | 0.1551 |
| English SMS SVM | 796 grouped UCI dev messages; spam class | 1.0000 | 0.9626 | 0.9810 |

SMS dev counts are TP=103, FP=0, FN=4, TN=689. It still misses some scams,
including the reported grant/SMS example that the separate rules flag.
CTI exact extraction remains limited. Hindi/Hinglish have functional regression
checks, **no independent learned accuracy estimate**. CTI/SMS test partitions
remain reserved; no single combined or final-test accuracy is claimed.

[Models and results](docs/MODELS_AND_RESULTS.md) explains active configuration,
metrics, limitations and held experiments. See [SMS provenance](docs/SMS_DATASET.md)
and [evaluation protocol](docs/paper_protocol.md) for split controls.

## Project structure

```text
src/cti/          Pipeline, NLP modules, models, training, evaluation and API
src/cti/web/      Python server and browser assets
configs/         Pipeline order, classical training and experiment settings
data/            CTI corpus, BIO data, split manifests and dictionaries
examples/        Authored demonstrations and regression cases
scripts/         Launcher, optional browser QA and experiment audits
tests/           Classical NLP, API, contract and split-lock tests
docs/            Setup, dataset provenance, model/results and review guides
checkpoints/     Development history; older suggestions may be superseded
results/         Compact reports; generated logs/screenshots stay local
models/          Ignored artifacts, downloaded SMS data and caches
```

Use [the documentation index](docs/README.md) and
[current status](checkpoints/CURRENT_STATUS.md). Historical neural experiments
remain inactive. `src/cti/core/model_policy.py` restricts NER to CRF, MEMM and
token Logistic Regression. spaCy provides blank tokenization, shapes and rule
matching, with no trained NER/parser.

## Validation and contribution

Run tests after changes. Optional Chrome QA checks real models, exports, source
offsets, error handling and mobile layout; see [setup](docs/SETUP.md#verification).
Actual local evidence is in [the finalization report](results/finalization_v1/README.md).
A configured CI workflow is separate from a verified remote run.

Develop on train/dev; use synthetic fixtures for test-lock checks. Keep final
test partitions reserved during development. Do not commit binaries, private
messages, environments, screenshots or generated per-row predictions.
Root `learning/` is ignored for private presentation preparation; public docs
live in `docs/`. Review upstream dataset attribution/licensing before reuse.
