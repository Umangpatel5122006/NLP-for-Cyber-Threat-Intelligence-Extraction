"""Loopback-only HTTP API and static UI for the classical CTI review demo."""

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from cti.core.pipeline import Pipeline
from cti.core.registry import create
from cti.extractors.message_risk import RULE_VERSION


ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(__file__).with_name("static")
MAX_CHARACTERS = 100_000
MAX_BODY_BYTES = 1_048_576
MODES = {"ner": "Entities + IOCs", "ioc": "IOCs only"}
SETUP_COMMAND = "python scripts/start_workbench.py"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class ReviewError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _load_pipeline(mode, message_model_path=None):
    name = "config_ner.yaml" if mode == "ner" else "config.yaml"
    config = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    modules = []
    for step in config["pipeline"]:
        params = dict(step.get("params", {}))
        if step["module"] == "message_classifier" and message_model_path is not None:
            params["model_path"] = str(ROOT / message_model_path)
        if "model_path" in params:
            params["model_path"] = str(ROOT / params["model_path"])
        modules.append(create(step["module"], **params))
    # The basic IOC config stays usable without any statistical NLP imports.
    # Workbench modes additionally expose the optional short-message model.
    if not any(module.name == "message_classifier" for module in modules):
        modules.insert(len(modules) - 1, create("message_classifier", model_path=str(ROOT / (message_model_path or "models/message_spam.joblib"))))
    return Pipeline(modules)


class ReviewEngine:
    def __init__(self, ioc_only=False, loader=None, message_model_path=None):
        self.ioc_only = ioc_only
        self.loader = _load_pipeline if loader is None else loader
        if loader is None and message_model_path is not None:
            self.loader = lambda mode: _load_pipeline(mode, message_model_path)
        self._pipelines, self._failures = {}, {}
        self._lock = threading.RLock()

    def _pipeline(self, mode):
        if mode not in MODES:
            raise ReviewError(400, "invalid_mode", "Choose a supported analysis mode.")
        if mode == "ner" and self.ioc_only:
            raise ReviewError(503, "mode_disabled", "Entity models are disabled in this server. IOC mode is available.")
        if mode in self._failures:
            raise self._failures[mode]
        if mode not in self._pipelines:
            try:
                self._pipelines[mode] = self.loader(mode)
            except FileNotFoundError:
                self._failures[mode] = ReviewError(503, "models_missing", "Saved models are missing. Use IOC mode or train the entity models.")
            except (ImportError, OSError):
                self._failures[mode] = ReviewError(503, "runtime_unavailable", "This runtime cannot load the entity models. Use IOC mode or the prepared WSL environment.")
            except ValueError:
                self._failures[mode] = ReviewError(503, "models_incompatible", "Saved models do not match this runtime or their feature resources. Start in the training environment or retrain in this environment.")
            except Exception:
                self._failures[mode] = ReviewError(503, "models_unavailable", "This mode could not initialize. Check the server setup and restart it.")
            if mode in self._failures:
                raise self._failures[mode]
        return self._pipelines[mode]

    def status(self):
        modes = []
        with self._lock:
            # Refresh must recover after a missing model is trained or a
            # transient import failure is fixed. Keep successful models cached.
            self._failures.clear()
            for mode, label in MODES.items():
                details = {"id": mode, "label": label, "ready": True, "message": ""}
                try:
                    self._pipeline(mode)
                except ReviewError as error:
                    details.update(ready=False, code=error.code, message=error.message)
                    if mode == "ner":
                        details["setup"] = {
                            "models_missing": ["Stop the old server with Ctrl+C, then run from the project root:", SETUP_COMMAND + " --train-models", "Reopen the app when the launcher starts the server."],
                            "runtime_unavailable": ["Stop the old server with Ctrl+C, then run from the project root:", SETUP_COMMAND, "The launcher checks native Python and the prepared WSL environment. See docs/WSL_SETUP.md to prepare WSL."],
                            "models_incompatible": ["Stop the old server with Ctrl+C, then run from the project root:", SETUP_COMMAND, "If neither runtime matches, use " + SETUP_COMMAND + " --train-models to retrain on train/dev in the chosen runtime."],
                            "mode_disabled": ["Restart the server without --ioc-only, or run " + SETUP_COMMAND + "."],
                        }.get(error.code, ["Check the server terminal and docs/WSL_SETUP.md, then refresh the connection."])
                modes.append(details)
            message_model = {"ready": False, "message": "No pipeline is ready."}
            for pipeline in self._pipelines.values():
                for module in pipeline.modules:
                    if module.name == "message_classifier":
                        message_model = module.readiness(retry=True)
                        break
                if message_model.get("ready"):
                    break
        return {"schema_version": 1, "modes": modes, "max_characters": MAX_CHARACTERS,
                "review_schema_version": 1,
                "language_support": {"ner": ["en"], "sms_classifier": ["en"], "message_rules": ["en", "hi", "hi-en"],
                                     "ioc_extraction": "language_independent_patterns",
                                     "language_detection": "script_and_lexical_heuristic; manual override; not verified language identification"},
                "message_classifier": message_model,
                "message_risk": {"method": "classical_multilingual_rules", "rule_version": RULE_VERSION,
                                 "indic_rule_version": 1, "supported_languages": ["en", "hi", "hi-en"], "available_in": list(MODES)}}

    def analyze(self, payload):
        if not isinstance(payload, dict) or not {"text", "mode"} <= set(payload) or set(payload) - {"text", "mode", "language"}:
            raise ReviewError(400, "invalid_request", "Provide report text and an analysis mode.")
        language = payload.get("language", "auto")
        if not isinstance(language, str) or language not in ("auto", "en", "hi", "hi-en"):
            raise ReviewError(400, "invalid_language", "Choose Auto, English, Hindi or Hinglish.")
        text, mode = payload["text"], payload["mode"]
        if not isinstance(mode, str) or mode not in MODES:
            raise ReviewError(400, "invalid_mode", "Choose a supported analysis mode.")
        if not isinstance(text, str) or not text.strip():
            raise ReviewError(400, "empty_report", "Add report text before analyzing.")
        if len(text) > MAX_CHARACTERS:
            raise ReviewError(413, "report_too_large", f"Reports can contain up to {MAX_CHARACTERS:,} characters.")
        # Reject unpaired surrogate escapes, which cannot be exported as UTF-8.
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise ReviewError(400, "invalid_unicode", "The report must contain valid UTF-8 text.") from None
        if not self._lock.acquire(blocking=False):
            raise ReviewError(429, "analysis_busy", "Another analysis is running. Try again shortly.")
        try:
            pipeline = self._pipeline(mode)
            started = time.perf_counter()
            document = pipeline.run(text, language_hint=language) if language != "auto" else pipeline.run(text)
            return {
                "document": asdict(document),
                "analysis": {"mode": mode, "elapsed_ms": round((time.perf_counter() - started) * 1000)},
            }
        finally:
            self._lock.release()


class ReviewHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, format, *args):
        # Reports and request details are not written to an access log.
        pass

    def _reply(self, status, body, content_type="application/json; charset=utf-8", head=False):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _check_origin(self):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host", "").lower()
        if host not in hosts:
            raise ReviewError(403, "invalid_host", "Use the server's localhost address.")
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + host:
            raise ReviewError(403, "invalid_origin", "Open the review app on the same local server.")

    def _error(self, error, head=False):
        self._reply(error.status, {"error": {"code": error.code, "message": error.message}}, head=head)

    def do_GET(self, head=False):
        try:
            self._check_origin()
            path = urlsplit(self.path).path
            if path in STATIC:
                filename, content_type = STATIC[path]
                self._reply(200, (ASSETS / filename).read_bytes(), content_type, head=head)
            elif path == "/api/status":
                self._reply(200, self.server.engine.status(), head=head)
            elif path == "/api/example":
                text = (ROOT / "examples/merge_report.txt").read_bytes().decode("utf-8")
                self._reply(200, {"text": text, "filename": "merge_report.txt"}, head=head)
            else:
                raise ReviewError(404, "not_found", "This page or endpoint was not found.")
        except ReviewError as error:
            self._error(error, head=head)

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_POST(self):
        try:
            self._check_origin()
            if urlsplit(self.path).path != "/api/analyze":
                raise ReviewError(404, "not_found", "This endpoint was not found.")
            if self.headers.get_content_type() != "application/json":
                raise ReviewError(415, "invalid_content_type", "Send a JSON analysis request.")
            length = self.headers.get("Content-Length")
            if length is None:
                raise ReviewError(411, "length_required", "The request needs a content length.")
            try:
                length = int(length)
            except ValueError:
                raise ReviewError(400, "invalid_length", "Invalid request length.") from None
            if not 0 < length <= MAX_BODY_BYTES:
                raise ReviewError(413, "request_too_large", "The analysis request is too large.")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise ReviewError(400, "invalid_json", "Send valid UTF-8 JSON.") from None
            self._reply(200, self.server.engine.analyze(payload))
        except ReviewError as error:
            self._error(error)
        except Exception as error:
            print(f"Analysis failed ({type(error).__name__}).", file=sys.stderr, flush=True)
            self._error(ReviewError(500, "analysis_failed", "Analysis failed. Check the local server terminal."))

    def do_OPTIONS(self):
        self._reply(405, {"error": {"code": "method_not_allowed", "message": "Use the local review app."}})


def create_server(host="127.0.0.1", port=8765, engine=None):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("The review demo binds to loopback only.")
    server = ThreadingHTTPServer((host, port), ReviewHandler)
    server.daemon_threads = True
    server.engine = ReviewEngine() if engine is None else engine
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ioc-only", action="store_true", help="Skip loading saved entity/action models.")
    parser.add_argument("--check", action="store_true", help="Print readiness JSON and exit; require NER unless --ioc-only is set.")
    parser.add_argument("--message-model", type=Path, help="Preview a trusted local SMS candidate without replacing the default model.")
    args = parser.parse_args(argv)
    engine = ReviewEngine(ioc_only=args.ioc_only, message_model_path=args.message_model)
    if args.check:
        status = engine.status()
        print(json.dumps(status), flush=True)
        required = "ioc" if args.ioc_only else "ner"
        raise SystemExit(0 if any(m["id"] == required and m["ready"] for m in status["modes"]) else 2)
    with create_server(args.host, args.port, engine) as server:
        print(f"CTI Workbench: http://{args.host}:{server.server_port}", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
