import copy
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from dataclasses import asdict
from pathlib import Path

from cti.core.base import BaseModule, Document, Entity
from cti.core.offsets import MappedText
from cti.core.pipeline import Pipeline
from cti.extractors.ioc_regex import IOCRegex
from cti.preprocessing.cleaner import TextCleaner
from cti.preprocessing.defanger import Defanger
from cti.preprocessing.sentence_splitter import SentenceSplitter


ROOT = Path(__file__).resolve().parents[1]


def normalized(raw):
    return Defanger().process(TextCleaner().process(Document(raw)))


class OffsetMappingTests(unittest.TestCase):
    def assert_original(self, document, text, original):
        start = document.text.index(text)
        span = document.original_span(start, start + len(text))
        self.assertEqual(document.original_text[slice(*span)], original)
        return span

    def test_identity_positions_and_non_bmp_characters_use_code_points(self):
        document = Document("\U0001f6e1 APT29")
        self.assertEqual(document.original_span(2, 7), (2, 7))
        self.assertEqual(document.original_text, "\U0001f6e1 APT29")

    def test_whitespace_and_deleted_characters_keep_their_original_evidence(self):
        raw = "\ufeff  Mimi\u200bkatz\t \r\nuses  malware. \n"
        document = normalized(raw)
        self.assertEqual(document.text, "Mimikatz uses malware.")
        self.assertEqual(document.original_text, raw)
        self.assert_original(document, "Mimikatz", "Mimi\u200bkatz")
        self.assert_original(document, "Mimikatz uses", "Mimi\u200bkatz\t \r\nuses")
        self.assert_original(document, "malware", "malware")

    def test_defang_rules_compose_back_to_the_input_report(self):
        pairs = [
            ("hxxps[://]evil[.]example(.)com/a", "https://evil.example.com/a"),
            ("HXXPS://evil{dot}com", "httpS://evil.com"),
            ("admin[at]evil[dot]com", "admin@evil.com"),
            ("admin[@]evil{ . }com", "admin@evil.com"),
            ("hxxp[:]//evil[.]com", "http://evil.com"),
            ("185.22.10[.]5", "185.22.10.5"),
        ]
        for original, expected in pairs:
            with self.subTest(original=original):
                document = normalized(" \t" + original + " \r\n")
                self.assertEqual(document.text, expected)
                self.assert_original(document, expected, original)

    def test_repeated_equal_indicators_map_to_distinct_original_occurrences(self):
        raw = "  192.0.2[.]1 then 192.0.2(.)1  "
        document = IOCRegex().process(normalized(raw))
        self.assertEqual(len(document.entities), 2)
        self.assertEqual([e.text for e in document.entities], ["192.0.2.1"] * 2)
        self.assertEqual([e.original_text for e in document.entities], ["192.0.2[.]1", "192.0.2(.)1"])
        self.assertEqual([e.original_start for e in document.entities], [raw.index("192"), raw.rindex("192")])

    def test_unicode_composition_tracks_multiple_input_characters(self):
        raw = "  Cafe\u0301 and \u1100\u1161\u11a8  "
        document = normalized(raw)
        self.assertEqual(document.text, "Caf\u00e9 and \uac01")
        self.assert_original(document, "\u00e9", "e\u0301")
        self.assert_original(document, "\uac01", "\u1100\u1161\u11a8")

    def test_unicode_reordering_uses_a_covering_span(self):
        # The grave composes across the earlier higher-class combining mark.
        document = normalized("a\u0315\u0300")
        self.assertEqual(document.text, "\u00e0\u0315")
        self.assert_original(document, "\u00e0", "a\u0315\u0300")
        self.assert_original(document, "\u0315", "\u0315")
        self.assertEqual(document.original_span(0, 2), (0, 3))

    def test_composition_and_uncomposed_mark_keep_distinct_sources(self):
        document = normalized("A\u0301\u0323")
        self.assertEqual(document.text, "\u1ea0\u0301")
        self.assert_original(document, "\u1ea0", "A\u0301\u0323")
        self.assert_original(document, "\u0301", "\u0301")

    def test_unicode_expansion_can_share_one_original_character(self):
        document = normalized("\u0344")
        self.assertEqual(document.text, "\u0308\u0301")
        self.assertEqual(document.original_span(0, 1), (0, 1))
        self.assertEqual(document.original_span(1, 2), (0, 1))

    def test_indic_joiners_are_preserved(self):
        raw = "\u0915\u094d\u200d\u0937 \u0a95\u0acd\u200c\u0ab7"
        document = normalized(raw)
        self.assertEqual(document.text, unicodedata.normalize("NFC", raw))
        self.assertEqual(document.original_span(0, len(document.text)), (0, len(raw)))

    def test_empty_and_fully_removed_text_keep_the_supplied_report(self):
        for raw in ("", " \r\n\ufeff\u200b\t"):
            with self.subTest(raw=raw):
                document = Pipeline.from_config(ROOT / "configs/config.yaml").run(raw)
                self.assertEqual(document.text, "")
                self.assertEqual(document.original_text, raw)
                self.assertEqual(document.entities, [])
                self.assertEqual(document.metadata["sentences"], [])

    def test_span_validation_rejects_invalid_and_empty_ranges(self):
        document = normalized(" APT29 ")
        for start, end in ((-1, 2), (0, 6), (2, 1), (0, 0), (True, 2), (0, 1.5)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    document.original_span(start, end)
        with self.assertRaisesRegex(ValueError, "zero-width"):
            MappedText.from_text("a").sub(re.compile(r"^"), "x")

    def test_untracked_changes_are_rejected(self):
        document = Document("APT29")
        document.text = "APT28"
        with self.assertRaisesRegex(ValueError, "without offset tracking"):
            document.original_span(0, 5)
        with self.assertRaisesRegex(ValueError, "without offset tracking"):
            TextCleaner().process(document)
        document = Document("APT29")
        document.original_text = "APT28"
        with self.assertRaisesRegex(ValueError, "without offset tracking"):
            document.attach_original_offsets()

    def test_text_changes_after_analysis_are_rejected_without_mutation(self):
        for with_entities in (True, False):
            document = Document(" APT29 ")
            if with_entities:
                document.entities.append(Entity("APT29", "TEST", 1, 6))
            else:
                SentenceSplitter().process(document)
            before = copy.deepcopy(asdict(document))
            with self.assertRaisesRegex(ValueError, "before sentence splitting"):
                TextCleaner().process(document)
            self.assertEqual(asdict(document), before)
            self.assertEqual(document.original_span(1, 6), (1, 6))

    def test_unchanged_transforms_are_idempotent_after_analysis(self):
        document = Pipeline.from_config(ROOT / "configs/config.yaml").run(" 185.22.10[.]5 ")
        before = copy.deepcopy(asdict(document))
        TextCleaner().process(document)
        Defanger().process(document)
        document.attach_original_offsets()
        self.assertEqual(asdict(document), before)

    def test_projection_validates_all_evidence_before_mutation(self):
        document = normalized(" APT29 ")
        document.entities = [Entity("APT29", "TEST", 0, 5), Entity("wrong", "TEST", 0, 5)]
        with self.assertRaisesRegex(ValueError, "Evidence text"):
            document.attach_original_offsets()
        self.assertIsNone(document.entities[0].original_start)
        self.assertNotIn("offsets", document.metadata)
        document.entities.pop()
        document.metadata["sentences"] = [{"text": "wrong", "start": 0, "end": 5}]
        with self.assertRaisesRegex(ValueError, "Evidence text"):
            document.attach_original_offsets()
        self.assertIsNone(document.entities[0].original_start)

    def test_pipeline_rejects_modules_that_discard_the_original_report(self):
        class UntrackedModule(BaseModule):
            def process(self, document):
                document.text = document.text.strip()
                return document

        class ResetModule(BaseModule):
            def process(self, document):
                return Document(document.text.strip())

        for module in (UntrackedModule(), ResetModule()):
            with self.subTest(module=type(module).__name__):
                with self.assertRaises(ValueError):
                    Pipeline([module]).run(" APT29 ")

    def test_pipeline_entities_and_sentences_include_original_evidence(self):
        raw = "\ufeff  APT29 contacted hxxps://evil[.]example[.]com.\r\nIt used 185.22.10[.]5.  "
        document = Pipeline.from_config(ROOT / "configs/config.yaml").run(raw)
        self.assertTrue(document.entities)
        for finding in [asdict(e) for e in document.entities] + document.metadata["sentences"]:
            self.assertEqual(document.text[finding["start"]:finding["end"]], finding["text"])
            self.assertEqual(raw[finding["original_start"]:finding["original_end"]], finding["original_text"])
        url = next(e for e in document.entities if e.label == "URL")
        self.assertEqual(url.original_text, "hxxps://evil[.]example[.]com")
        self.assertEqual(document.metadata["sentences"][0]["original_start"], raw.index("APT29"))
        self.assertEqual(document.metadata["offsets"]["unit"], "unicode_code_points")
        self.assertEqual(document.metadata["offsets"]["interval"], "half_open")
        payload = asdict(document)
        self.assertEqual(payload["original_text"], raw)
        self.assertFalse(any(key.startswith("_") for key in payload))

    def test_cli_preserves_file_bytes_as_unicode_including_crlf_and_bom(self):
        raw = "\ufeff\U0001f6e1  Contact 185.22.10[.]5.\r\nAnother line.\r\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            path.write_bytes(raw.encode("utf-8"))
            output = subprocess.check_output([
                sys.executable, "-m", "cti.analyze", "--config", str(ROOT / "configs/config.yaml"),
                "--input", str(path),
            ], cwd=ROOT, text=True, encoding="utf-8", timeout=30)
        payload = json.loads(output)
        self.assertEqual(payload["original_text"], raw)
        ip = next(e for e in payload["entities"] if e["label"] == "IP")
        self.assertEqual(ip["original_start"], raw.index("185"))
        self.assertEqual(ip["original_text"], "185.22.10[.]5")


if __name__ == "__main__":
    unittest.main()
