import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("workbench_launcher", Path(__file__).resolve().parents[1] / "scripts/start_workbench.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def status(ner=False, ioc=True):
    return {"modes": [{"id": "ner", "ready": ner}, {"id": "ioc", "ready": ioc}]}


class LauncherTests(unittest.TestCase):
    def test_selects_compatible_runtime_instead_of_ioc_only_fallback(self):
        with patch.object(launcher, "candidates", return_value=[("native", ["native"]), ("WSL", ["wsl"])]), \
             patch.object(launcher, "readiness", side_effect=[status(), status(ner=True)]), \
             patch.object(launcher.subprocess, "call", return_value=0) as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                launcher.main(["--port", "8770"])
            self.assertEqual(raised.exception.code, 0)
            self.assertEqual(run.call_args.args[0], ["wsl", "-m", "cti.web", "--port", "8770"])

    def test_no_implicit_training_when_models_are_missing(self):
        with patch.object(launcher, "candidates", return_value=[("native", ["native"])]), \
             patch.object(launcher, "readiness", return_value=status()), \
             patch.object(launcher.subprocess, "call") as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                launcher.main([])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()

    def test_check_never_trains_or_starts_even_with_training_option(self):
        with patch.object(launcher, "candidates", return_value=[("native", ["native"])]), \
             patch.object(launcher, "readiness", return_value=status()), \
             patch.object(launcher.subprocess, "call") as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                launcher.main(["--check", "--train-models"])
            run.assert_not_called()

    def test_explicit_ioc_mode_skips_search_for_ner_runtime(self):
        with patch.object(launcher, "candidates", return_value=[("native", ["native"])]), \
             patch.object(launcher, "readiness", return_value=status()) as check, \
             patch.object(launcher.subprocess, "call") as run, contextlib.redirect_stdout(io.StringIO()):
            launcher.main(["--ioc-only", "--check"])
            check.assert_called_once_with(["native"], True)
            run.assert_not_called()

    def test_explicit_training_skips_a_runtime_with_blocked_native_imports(self):
        with patch.object(launcher, "candidates", return_value=[("blocked", ["blocked"]), ("working", ["working"])]), \
             patch.object(launcher, "readiness", side_effect=[status(), status(), status(ner=True)]), \
             patch.object(launcher.subprocess, "run", side_effect=[type("Probe", (), {"returncode": 1})(), type("Probe", (), {"returncode": 0})()]), \
             patch.object(launcher.subprocess, "call", return_value=0) as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                launcher.main(["--train-models"])
            self.assertEqual(run.call_args_list[0].args[0], ["working", "-m", "cti.train_ner"])
            self.assertEqual(len(run.call_args_list), 2)

    def test_explicit_sms_training_reloads_model_before_starting(self):
        missing, available = status(ner=True), status(ner=True)
        missing["message_classifier"] = {"ready": False}
        available["message_classifier"] = {"ready": True}
        with patch.object(launcher, "candidates", return_value=[("working", ["working"])]), \
             patch.object(launcher, "readiness", side_effect=[missing, available]), \
             patch.object(launcher.subprocess, "call", return_value=0) as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                launcher.main(["--train-message-model"])
            self.assertEqual(run.call_args_list[0].args[0], ["working", "-m", "cti.train_message", "--download"])
            self.assertEqual(run.call_args_list[1].args[0], ["working", "-m", "cti.web", "--port", "8765"])

    def test_check_does_not_train_missing_sms_even_when_requested(self):
        missing = status(ner=True); missing["message_classifier"] = {"ready": False}
        with patch.object(launcher, "candidates", return_value=[("working", ["working"])]), \
             patch.object(launcher, "readiness", return_value=missing), \
             patch.object(launcher.subprocess, "call") as run, contextlib.redirect_stdout(io.StringIO()):
            launcher.main(["--train-message-model", "--check"])
            run.assert_not_called()

    def test_ioc_mode_prefers_runtime_with_loaded_message_classifier(self):
        missing, available = status(), status()
        missing["message_classifier"] = {"ready": False}
        available["message_classifier"] = {"ready": True}
        with patch.object(launcher, "candidates", return_value=[("native", ["native"]), ("WSL", ["wsl"])]), \
             patch.object(launcher, "readiness", side_effect=[missing, available]), \
             patch.object(launcher.subprocess, "call", return_value=0) as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                launcher.main(["--ioc-only"])
            self.assertEqual(run.call_args.args[0], ["wsl", "-m", "cti.web", "--port", "8765", "--ioc-only"])

    def test_candidate_preview_path_is_forwarded_to_check_and_server(self):
        available = status(ner=True); available["message_classifier"] = {"ready": True}
        with patch.object(launcher, "candidates", return_value=[("working", ["working"])]), \
             patch.object(launcher, "readiness", return_value=available) as check, \
             patch.object(launcher.subprocess, "call", return_value=0) as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                launcher.main(["--message-model", "models/candidate.joblib", "--port", "8766"])
            check.assert_called_once_with(["working"], False, "models/candidate.joblib")
            self.assertEqual(run.call_args.args[0][-2:], ["--message-model", "models/candidate.joblib"])

    def test_unavailable_explicit_candidate_does_not_start_a_fallback_server(self):
        unavailable = status(ner=True); unavailable["message_classifier"] = {"ready": False}
        with patch.object(launcher, "candidates", return_value=[("working", ["working"])]), \
             patch.object(launcher, "readiness", return_value=unavailable), \
             patch.object(launcher.subprocess, "call") as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                launcher.main(["--message-model", "models/missing.joblib"])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
