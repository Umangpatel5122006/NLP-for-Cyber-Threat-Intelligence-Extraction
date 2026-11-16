"""Start the local workbench in a compatible runtime, without changing policy.

Run from the project root: python scripts/start_workbench.py
Use --train-models to train missing/incompatible artifacts on train/dev only.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def wsl_candidates(distro=None, python_path=None):
    """Pass paths as arguments, never as shell source; use WSL's default distro."""
    base = ["wsl", *(["-d", distro] if distro else []), "--exec", "bash", "-c"]
    label = distro or "default distribution"
    yield f"WSL ({label})", [*base, 'exec "$@"', "cti-launch", python_path or ".venv-wsl/bin/python"]
    if python_path is None:
        # Keep existing checkouts working without requiring their environment
        # to move. New installations use the documented project environment.
        legacy_base = ["wsl", "-d", distro or "Ubuntu-24.04", "--exec", "bash", "-c"]
        yield "prepared Ubuntu WSL", [*legacy_base,
            'exec "$HOME/.cache/cti-classical-py312/bin/python" "$@"', "cti-launch"]


def candidates(wsl_distro=None, wsl_python=None):
    local = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    seen = set()
    for name, executable in (("project .venv", local), ("current Python", Path(sys.executable))):
        key = os.path.normcase(str(executable.resolve()))
        if executable.is_file() and key not in seen:
            seen.add(key)
            yield name, [str(executable)]
    if os.name == "nt":
        # WSL inherits the checkout cwd. No Windows security policy changes.
        yield from wsl_candidates(wsl_distro, wsl_python)


def readiness(command, ioc_only=False, message_model=None):
    try:
        result = subprocess.run(
            [*command, "-m", "cti.web", "--check", *(["--ioc-only"] if ioc_only else []), *(["--message-model", str(message_model)] if message_model else [])],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
        )
        # Exit 2 is the documented "requested mode unavailable" check result.
        # A crashed interpreter can print JSON before failing during teardown.
        if result.returncode not in (0, 2):
            return None
        return json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def ready(status, mode):
    return bool(status and any(m["id"] == mode and m["ready"] for m in status.get("modes", [])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ioc-only", action="store_true")
    parser.add_argument("--train-models", action="store_true", help="Train on train/dev only if no compatible saved NER models are found.")
    parser.add_argument("--train-message-model", action="store_true", help="Download the pinned SMS corpus and train the classical SMS model when unavailable.")
    parser.add_argument("--check", action="store_true", help="Check setup without starting a server or training.")
    parser.add_argument("--message-model", type=Path, help="Preview a trusted candidate artifact; does not modify default model files.")
    parser.add_argument("--wsl-distro", help="WSL distribution to try on Windows; otherwise use the default distribution.")
    parser.add_argument("--wsl-python", help="Linux Python path for the WSL candidate; default: .venv-wsl/bin/python (absolute paths also accepted).")
    args = parser.parse_args(argv)
    preview_path = None
    if args.message_model:
        try:
            preview_path = (ROOT / args.message_model).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            parser.error("The portable launcher requires a candidate inside this project; use python -m cti.web directly for other paths.")
    if args.message_model and (args.train_models or args.train_message_model):
        parser.error("Train candidates with cti.experiment_message, then preview without training flags.")
    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")
    required = "ioc" if args.ioc_only else "ner"
    checked, chosen, fallback = [], None, None
    for name, command in candidates(args.wsl_distro, args.wsl_python):
        print(f"Checking {name}...", flush=True)
        status = readiness(command, args.ioc_only, preview_path) if preview_path else readiness(command, args.ioc_only)
        checked.append((name, command, status))
        if ready(status, required):
            if args.message_model and not status.get("message_classifier", {}).get("ready"):
                print(status.get("message_classifier", {}).get("message", "The requested preview model is unavailable."), flush=True)
                continue
            if status.get("message_classifier") and not status["message_classifier"].get("ready"):
                fallback = fallback or (name, command, status)
                if (args.train_models or args.train_message_model) and not args.check:
                    try:
                        probe = subprocess.run([*command, "-c", "import sklearn.svm"], cwd=ROOT, capture_output=True, timeout=90)
                    except (OSError, subprocess.TimeoutExpired):
                        continue
                    if probe.returncode == 0:
                        chosen = (name, command, status)
                        break
                continue
            chosen = (name, command, status)
            break
        if status:
            mode = next((m for m in status["modes"] if m["id"] == required), {})
            print(mode.get("message", "Mode unavailable."), flush=True)
        else:
            print("Project dependencies could not initialize in this runtime.", flush=True)
    if chosen is None and args.train_models and not args.check and not args.ioc_only:
        for name, command, status in checked:
            if not ready(status, "ioc"):
                continue
            # An IOC-capable interpreter can still have blocked native NLP
            # imports. Probe the actual classical tokenizer before training.
            try:
                probe = subprocess.run(
                    [*command, "-c", "import sklearn_crfsuite; from cti.preprocessing.tokenizer import tokenize; tokenize('check')"],
                    cwd=ROOT, capture_output=True, timeout=90,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if probe.returncode:
                continue
            print(f"Training configured classical models in {name} (train/dev only)...", flush=True)
            training_code = subprocess.call([*command, "-m", "cti.train_ner"], cwd=ROOT)
            if training_code:
                raise SystemExit(f"Training failed (process exit code {training_code}). Check the terminal output; the server was not started.")
            status = readiness(command)
            if ready(status, "ner"):
                chosen = (name, command, status)
                break
    chosen = chosen or fallback
    if chosen is None:
        print("Full entity mode is unavailable." if required == "ner" else "IOC mode is unavailable.")
        print("Install the pinned dependencies using docs/SETUP.md; optional WSL setup is in docs/WSL_SETUP.md.")
        if required == "ner":
            print("For missing/incompatible models: python scripts/start_workbench.py --train-models")
            print("To use IOCs and scam-pattern checks: python scripts/start_workbench.py --ioc-only")
        raise SystemExit(2)
    name, command, status = chosen
    print(f"Selected {name}: {required} mode ready.", flush=True)
    if not args.check and (args.train_models or args.train_message_model) and not status.get("message_classifier", {}).get("ready", True):
        print("Training the classical SMS model (grouped train CV, dev only; no test evaluation)...", flush=True)
        training_code = subprocess.call([*command, "-m", "cti.train_message", "--download"], cwd=ROOT)
        if training_code:
            raise SystemExit(f"SMS training failed (process exit code {training_code}). Check the terminal output; the server was not started.")
        status = readiness(command, args.ioc_only)
        if not status or not status.get("message_classifier", {}).get("ready"):
            raise SystemExit("SMS model could not reload after training.")
    if status.get("message_classifier") and not status["message_classifier"].get("ready"):
        print("SMS model unavailable: rules/IOCs remain available. Use --train-message-model in a working ML runtime.", flush=True)
    if args.check:
        print(json.dumps(status), flush=True)
        return
    try:
        raise SystemExit(subprocess.call(
            [*command, "-m", "cti.web", "--port", str(args.port), *(["--ioc-only"] if args.ioc_only else []), *(["--message-model", preview_path] if preview_path else [])], cwd=ROOT,
        ))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
