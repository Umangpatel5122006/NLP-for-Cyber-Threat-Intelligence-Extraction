import subprocess
import sys
import unittest
from pathlib import Path

from cti.core.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_configured_pipeline_extracts_iocs_with_valid_offsets(self):
        pipeline = Pipeline.from_config(ROOT / "configs/config.yaml")
        document = pipeline.run(
            "  APT29 used   Mimikatz.  It contacted "
            "hxxps://evil[.]example[.]com and 185.22.10[.]5.  "
        )
        self.assertEqual(document.metadata["language"], "en")
        self.assertEqual(len(document.metadata["sentences"]), 2)
        findings = {(entity.label, entity.text) for entity in document.entities}
        self.assertIn(("URL", "https://evil.example.com"), findings)
        self.assertIn(("IP", "185.22.10.5"), findings)
        for entity in document.entities:
            self.assertEqual(document.text[entity.start:entity.end], entity.text)

    def test_pipeline_starts_without_unselected_nlp_libraries(self):
        # A fresh interpreter avoids registrations left over by other tests.
        code = (
            "import sys\n"
            "from cti.core.pipeline import Pipeline\n"
            "Pipeline.from_config('configs/config.yaml').run('IP: 10.1.2.3.')\n"
            "for name in ('spacy', 'pandas', 'sklearn', 'torch', 'transformers'):\n"
            "    assert name not in sys.modules, name\n"
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
