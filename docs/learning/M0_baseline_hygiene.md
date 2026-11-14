# Learning Note M0: Baseline Lock and Repository Hygiene

## 1. The Big Picture
In real-world Natural Language Processing (NLP) and Cyber Threat Intelligence (CTI) projects, baseline locking and repository hygiene are fundamental prerequisites before writing or tuning any machine learning models. 

Before optimizing hyperparameters or modifying feature extractors:
- **Baseline Lock**: You must record the exact performance metrics and test behavior of the existing system as-is. Without an immutable baseline record, it is impossible to prove whether later changes genuinely improved performance or introduced silent regressions.
- **Dependency Hygiene**: NLP pipelines often mix lightweight classical algorithms (regular expressions, linear classifiers, CRFs) with heavy deep learning frameworks (PyTorch, Hugging Face Transformers). Pinning dependencies prevents breaking changes when upstream packages update. Splitting core dependencies from optional GPU/deep learning dependencies ensures that a CPU-only researcher or production server can install and run the statistical pipeline in seconds without downloading gigabytes of CUDA binaries.

## 2. What We Changed and Why
In Milestone 0, we made zero behavioral code changes. Instead, we hardened the environment and recorded the baseline state:

1. **Created `requirements.txt`**:
   - Pinned exact versions for the statistical CTI pipeline: `spacy==3.8.16`, `scikit-learn==1.9.1`, `sklearn-crfsuite==0.5.0`, `seqeval==1.2.2`, `ioc-finder==9.4.1`, `pyyaml==6.0.3`, `pandas==3.1.0`, and `joblib==1.6.0`.
   - *Why*: Guarantees reproducibility across team members and CI environments using standard Python 3.13 without unexpected API changes.

2. **Created `requirements-optional.txt`**:
   - Pinned optional deep learning packages: `torch==2.14.1`, `pytorch-crf==0.7.2`, `transformers==5.20.2`, and `accelerate==1.15.0`.
   - *Why*: Keeps transformer dependencies isolated from the CPU-first statistical core, allowing the project to remain lightweight and portable.

3. **Populated `.gitignore`**:
   - Added patterns for virtual environments (`venv/`), Python bytecode caches (`__pycache__/`, `*.pyc`), build artifacts (`*.egg-info/`), trained model binaries (`models/`), and temporary run files (`results/*.tmp`). Untracked accidentally committed `.pyc` files from the repository index while keeping all raw and external datasets explicitly tracked as requested.
   - *Why*: Keeps git history clean, prevents merge conflicts from binary caches, and ensures model checkpoints do not bloat source version control.

4. **Captured `results/baseline_before_changes.txt`**:
   - Ran `python -m unittest discover -s tests` and `python notebooks/12_relevant_only.py --layer entities --models crf_tagger,memm_tagger --skip-relevance`.
   - *Why*: Establishes the unvarnished reference point for precision, recall, and F1 score against which all future milestones are verified.

## 3. Alternative Techniques (Classical vs Modern)

| Dimension | Classical Approach (What We Did) | Modern / Production Alternative |
|---|---|---|
| **Dependency Management** | Flat `requirements.txt` with exact version pins (`==`). | Lockfile managers such as **uv** (`uv pip compile` / `uv.lock`) or **Poetry** (`poetry.lock`). These resolve the full transitive dependency graph and compute cryptographic hashes for each wheel. |
| **Environment Containerization** | Local Python virtual environment (`virtualenv` / `venv`). | **Docker / OCI Containers**: Encapsulates the OS, C-libraries (e.g. `libspatialindex`, system regex, glibc), and Python environment into an immutable image. |
| **Experiment Tracking** | Plain text / markdown logs in a `results/` folder checked into source control. | Dedicated tracking platforms like **MLflow**, **Weights & Biases (W&B)**, or **DVC (Data Version Control)** for tracking metrics, hyperparameters, and large artifact lineage. |
| **Test Execution** | Python's standard `unittest` runner. | **pytest** with plugins (`pytest-xdist` for parallel execution, `pytest-cov` for coverage reporting). |

## 4. How to Swap in an Alternative (One-Line Config)
To migrate from standard `pip` to the ultra-fast modern package resolver `uv`:

1. Install `uv`:
   ```bash
   pip install uv
   ```
2. In one line, compile and lock all dependencies into a deterministic lockfile:
   ```bash
   uv pip compile pyproject.toml -o requirements.lock
   ```
3. Or install the core environment deterministically in a single command:
   ```bash
   uv pip install -r requirements.txt
   ```
