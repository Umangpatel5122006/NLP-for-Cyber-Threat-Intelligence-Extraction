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


if __name__ == "__main__":
    unittest.main()
