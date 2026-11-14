import contextlib
import importlib.util
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from cti.core.model_policy import CLASSICAL_NER_MODULES, EXCLUDED_NER_MODELS
from cti.core.registry import create, register


ROOT = Path(__file__).resolve().parents[1]


class ClassicalModelPolicyTests(unittest.TestCase):
    def test_active_ner_choices_are_classical(self):
        self.assertEqual(set(CLASSICAL_NER_MODULES), {"crf_tagger", "memm_tagger", "token_logreg"})
        self.assertFalse(set(CLASSICAL_NER_MODULES) & EXCLUDED_NER_MODELS)

    def test_registry_rejects_excluded_models(self):
        for name in EXCLUDED_NER_MODELS:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "deep learning"):
                    create(name)
                with self.assertRaisesRegex(ValueError, "deep learning"):
                    register(name)

    def test_historical_imports_fail_before_loading_neural_dependencies(self):
        code = (
            "import importlib, sys\n"
            f"names = {sorted(EXCLUDED_NER_MODELS)!r}\n"
            "for name in names:\n"
            "    try:\n"
            "        importlib.import_module('cti.models.' + name)\n"
            "    except ValueError as error:\n"
            "        assert 'deep learning' in str(error)\n"
            "    else:\n"
            "        raise AssertionError(name)\n"
            "for name in ('torch', 'transformers', 'spacy', 'torchcrf'):\n"
            "    assert name not in sys.modules, name\n"
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_comparison_scripts_reject_neural_choices_before_reading_data(self):
        for filename, mapping_name, loader_name in (
            ("07_compare.py", "MODELS", "load_bio"),
            ("12_relevant_only.py", "MODEL_MODULES", "load_smah"),
        ):
            spec = importlib.util.spec_from_file_location("comparison_under_test", ROOT / "notebooks" / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(set(getattr(module, mapping_name)), set(CLASSICAL_NER_MODULES))
            for name in sorted(EXCLUDED_NER_MODELS):
                with self.subTest(script=filename, name=name):
                    with patch.object(module, loader_name) as loader:
                        with patch.object(sys, "argv", [filename, "--models", name]):
                            with contextlib.redirect_stderr(io.StringIO()):
                                with self.assertRaises(SystemExit) as error:
                                    module.main()
                        self.assertEqual(error.exception.code, 2)
                        loader.assert_not_called()

    def test_comparison_defaults_contain_no_neural_models_or_epoch_option(self):
        path = ROOT / "notebooks/12_relevant_only.py"
        spec = importlib.util.spec_from_file_location("default_comparison_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parse_args = module.argparse.ArgumentParser.parse_args
        parsed = []

        def capture_args(parser, *args, **kwargs):
            namespace = parse_args(parser, *args, **kwargs)
            parsed.append(namespace)
            return namespace

        with patch.object(module, "load_smah", side_effect=RuntimeError("stop before data load")):
            with patch.object(module, "_make_model") as factory:
                with patch.object(module.argparse.ArgumentParser, "parse_args", capture_args):
                    with patch.object(sys, "argv", [str(path)]):
                        with self.assertRaisesRegex(RuntimeError, "stop before data load"):
                            module.main()
                factory.assert_not_called()
        self.assertEqual(set(parsed[0].models.split(",")), set(CLASSICAL_NER_MODULES))
        with patch.object(sys, "argv", [str(path), "--epochs", "1"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    module.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
