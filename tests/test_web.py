import contextlib
import http.client
import io
import json
import threading
import unittest
from unittest.mock import patch

from cti.web.server import MAX_BODY_BYTES, MAX_CHARACTERS, ReviewEngine, ReviewError, _load_pipeline, create_server, main
from test_message_risk import USER_MESSAGE


def available_iocs(mode):
    if mode == "ner":
        raise FileNotFoundError("synthetic missing artifact")
    return _load_pipeline(mode)


class ReviewEngineTests(unittest.TestCase):
    def test_status_reports_unavailable_models_and_keeps_ioc_mode_ready(self):
        status = ReviewEngine(loader=available_iocs).status()
        modes = {mode["id"]: mode for mode in status["modes"]}
        self.assertFalse(modes["ner"]["ready"])
        self.assertEqual(modes["ner"]["code"], "models_missing")
        self.assertTrue(modes["ioc"]["ready"])
        self.assertIn("--train-models", " ".join(modes["ner"]["setup"]))

    def test_readiness_retry_recovers_missing_models_without_restarting(self):
        repaired, calls = False, []

        def loader(mode):
            calls.append(mode)
            if mode == "ner" and not repaired:
                raise FileNotFoundError("missing")
            return _load_pipeline("ioc")

        engine = ReviewEngine(loader=loader)
        self.assertFalse(engine.status()["modes"][0]["ready"])
        repaired = True
        self.assertTrue(engine.status()["modes"][0]["ready"])
        self.assertEqual(calls, ["ner", "ioc", "ner"])
        self.assertEqual(engine.analyze({"mode": "ner", "text": USER_MESSAGE})["document"]["metadata"]["message_risk"]["status"], "suspicious")

    def test_runtime_and_artifact_errors_have_distinct_setup_instructions(self):
        for error, code in [(ImportError("blocked import"), "runtime_unavailable"), (ValueError("private manifest detail"), "models_incompatible")]:
            with self.subTest(code=code):
                def loader(mode):
                    if mode == "ner":
                        raise error
                    return _load_pipeline(mode)
                status = ReviewEngine(loader=loader).status()
                self.assertEqual(status["modes"][0]["code"], code)
                self.assertTrue(status["modes"][0]["setup"])
                self.assertNotIn("private manifest detail", json.dumps(status))
                self.assertTrue(status["modes"][1]["ready"])

    def test_check_cli_exits_with_required_mode_readiness(self):
        for argv, expected in [(["--check"], 2), (["--check", "--ioc-only"], 0)]:
            with self.subTest(argv=argv), patch("cti.web.server.ReviewEngine", return_value=ReviewEngine(loader=available_iocs)):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                    main(argv)
                self.assertEqual(raised.exception.code, expected)
                self.assertIn("modes", json.loads(output.getvalue()))

    def test_ioc_only_does_not_attempt_to_load_ner(self):
        loaded = []

        def loader(mode):
            loaded.append(mode)
            return _load_pipeline(mode)

        engine = ReviewEngine(ioc_only=True, loader=loader)
        engine.status()
        self.assertEqual(loaded, ["ioc"])
        with self.assertRaises(ReviewError) as raised:
            engine.analyze({"mode": "ner", "text": "APT29"})
        self.assertEqual(raised.exception.code, "mode_disabled")

    def test_request_validation_rejects_empty_large_and_invalid_unicode_reports(self):
        engine = ReviewEngine(ioc_only=True)
        for payload, code in [
            ({}, "invalid_request"), ([], "invalid_request"),
            ({"text": "APT29", "mode": "neural"}, "invalid_mode"),
            ({"text": " ", "mode": "ioc"}, "empty_report"),
            ({"text": 5, "mode": "ioc"}, "empty_report"),
            ({"text": "x" * (MAX_CHARACTERS + 1), "mode": "ioc"}, "report_too_large"),
            ({"text": "\ud800", "mode": "ioc"}, "invalid_unicode"),
        ]:
            with self.subTest(code=code):
                with self.assertRaises(ReviewError) as raised:
                    engine.analyze(payload)
                self.assertEqual(raised.exception.code, code)

    def test_engine_reuses_pipeline_without_leaking_prior_document_findings(self):
        engine = ReviewEngine(ioc_only=True)
        first = engine.analyze({"text": "Contact 185.22.10[.]5.", "mode": "ioc"})
        second = engine.analyze({"text": "Background information only.", "mode": "ioc"})
        self.assertTrue(first["document"]["findings"])
        self.assertEqual(second["document"]["findings"], [])
        self.assertEqual(len(engine._pipelines), 1)


class ReviewHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ReviewEngine(loader=available_iocs)
        cls.server = create_server(port=0, engine=cls.engine)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, path, method="GET", payload=None, body=None, headers=None):
        headers = dict(headers or {})
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(connection.close)
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()

    def test_static_app_assets_and_response_headers(self):
        for path, content_type in [("/", "text/html"), ("/app.js", "text/javascript"), ("/styles.css", "text/css")]:
            with self.subTest(path=path):
                status, headers, body = self.request(path)
                self.assertEqual(status, 200)
                self.assertTrue(headers["Content-Type"].startswith(content_type))
                self.assertTrue(body)
                self.assertNotIn(b"\x00", body)
                self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_head_returns_headers_without_body(self):
        status, headers, body = self.request("/", method="HEAD")
        self.assertEqual(status, 200)
        self.assertGreater(int(headers["Content-Length"]), 0)
        self.assertEqual(body, b"")

    def test_status_and_synthetic_example_are_available(self):
        status, _, body = self.request("/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["schema_version"], 1)
        status, _, body = self.request("/api/example")
        self.assertEqual(status, 200)
        self.assertIn("hxxps://", json.loads(body)["text"])
        self.assertEqual(json.loads(body)["filename"], "merge_report.txt")

    def test_real_ioc_api_preserves_unicode_crlf_bom_and_evidence(self):
        original = "\ufeff\U0001f6e1 Contact 185.22.10[.]5.\r\nNext line.\r\n"
        status, _, body = self.request("/api/analyze", "POST", {"mode": "ioc", "text": original})
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["analysis"]["mode"], "ioc")
        self.assertGreaterEqual(result["analysis"]["elapsed_ms"], 0)
        document = result["document"]
        self.assertEqual(document["original_text"], original)
        finding = next(item for item in document["findings"] if item["label"] == "IP")
        self.assertEqual(finding["original_start"], original.index("185"))
        self.assertEqual(finding["original_text"], "185.22.10[.]5")
        self.assertEqual(finding["evidence"][0]["source"], "ioc_finder")

    def test_report_markup_is_data_not_a_server_template(self):
        text = '<script>window.injected=true</script> Contact 185.22.10[.]5.'
        status, headers, body = self.request("/api/analyze", "POST", {"mode": "ioc", "text": text})
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertEqual(json.loads(body)["document"]["original_text"], text)

    def test_scam_check_is_available_when_ner_models_are_missing(self):
        status, _, body = self.request("/api/analyze", "POST", {"mode": "ioc", "text": USER_MESSAGE})
        self.assertEqual(status, 200)
        doc = json.loads(body)["document"]
        self.assertEqual(doc["metadata"]["message_risk"]["status"], "suspicious")
        self.assertEqual(doc["findings"], [])
        self.assertEqual(doc["original_text"], USER_MESSAGE)

    def test_unknown_paths_and_traversal_do_not_serve_repository_files(self):
        for path in ("/AGENTS.md", "/../AGENTS.md", "/%2e%2e/AGENTS.md", "/api/example?file=data/raw"):
            with self.subTest(path=path):
                status, _, body = self.request(path)
                if path.startswith("/api/example?"):
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body)["filename"], "merge_report.txt")
                else:
                    self.assertEqual(status, 404)

    def test_cross_origin_and_foreign_host_requests_are_rejected(self):
        for headers in [{"Origin": "https://example.com"}, {"Origin": "null"}, {"Host": "example.com"}]:
            with self.subTest(headers=headers):
                status, _, _ = self.request("/api/analyze", "POST", {"mode": "ioc", "text": "APT29"}, headers=headers)
                self.assertEqual(status, 403)

    def test_json_content_type_and_body_validation(self):
        cases = [
            ({"Content-Type": "text/plain"}, b"{}", 415),
            ({"Content-Type": "application/json"}, b"invalid", 400),
            ({"Content-Type": "application/json"}, b"\xff", 400),
            ({"Content-Type": "application/json"}, b"[]", 400),
            ({"Content-Type": "application/json"}, b'{"text":"","mode":"ioc"}', 400),
            ({"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)}, b"{}", 413),
        ]
        for headers, body, expected in cases:
            with self.subTest(expected=expected, body=body):
                status, _, _ = self.request("/api/analyze", "POST", body=body, headers=headers)
                self.assertEqual(status, expected)

    def test_missing_content_length_is_rejected(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(connection.close)
        connection.putrequest("POST", "/api/analyze")
        connection.putheader("Content-Type", "application/json")
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 411)

    def test_missing_models_return_an_actionable_503(self):
        status, _, body = self.request("/api/analyze", "POST", {"mode": "ner", "text": "APT29"})
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"]["code"], "models_missing")

    def test_analysis_busy_is_reported_without_starting_another_run(self):
        self.engine._lock.acquire()
        try:
            status, _, body = self.request("/api/analyze", "POST", {"mode": "ioc", "text": "APT29"})
        finally:
            self.engine._lock.release()
        self.assertEqual(status, 429)
        self.assertEqual(json.loads(body)["error"]["code"], "analysis_busy")

    def test_unexpected_analysis_error_does_not_leak_exception_text(self):
        with contextlib.redirect_stderr(io.StringIO()), patch.object(self.engine, "analyze", side_effect=RuntimeError("private implementation detail")):
            status, _, body = self.request("/api/analyze", "POST", {"mode": "ioc", "text": "APT29"})
        self.assertEqual(status, 500)
        self.assertNotIn(b"private implementation detail", body)
        self.assertEqual(json.loads(body)["error"]["code"], "analysis_failed")

    def test_server_rejects_non_loopback_binding(self):
        with self.assertRaises(ValueError):
            create_server("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
