import json
import os
import tempfile
import unittest
from pathlib import Path

from cti.core.base import Document
from cti.core.bio import to_bio
from cti.core.dataset import load_smah
from cti.models.base import validate_training_data

from cti.models.crf import CRFTagger
from cti.models.data import load_bio
from cti.models.memm import MEMMTagger, TokenLogisticTagger
from cti.preprocessing.cleaner import TextCleaner
from cti.preprocessing.defanger import Defanger
from cti.preprocessing.language_detector import LanguageDetector
from cti.preprocessing.sentence_splitter import SentenceSplitter
from cti.preprocessing.tokenizer import tokenize


class ModelContractTests(unittest.TestCase):
    def test_training_data_checks_alignment_and_bio_transitions(self):
        validate_training_data([["apt29", "used"]], [["B-ACTOR", "O"]])
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_training_data([["apt29"]], [[]])
        with self.assertRaisesRegex(ValueError, "orphaned I-tag"):
            validate_training_data([["apt29"]], [["I-ACTOR"]])

    def test_crf_preserves_empty_sentence_positions(self):
        model = CRFTagger(max_iterations=5)
        model.fit(
            [["apt29", "used"], ["mimikatz"]],
            [["B-ACTOR", "O"], ["B-TOOL"]],
        )
        predictions = model.predict([["apt29"], [], ["mimikatz", "ran"]])
        self.assertEqual([len(sequence) for sequence in predictions], [1, 0, 2])

    def test_sklearn_sequence_taggers_follow_bio_interface(self):
        tokens = [
            ["apt29", "used", "mimikatz"],
            ["lazarus", "used", "malware"],
        ]
        tags = [
            ["B-THREAT_ACTOR", "O", "B-MALWARE_TOOL"],
            ["B-THREAT_ACTOR", "O", "B-MALWARE_TOOL"],
        ]
        for model in (MEMMTagger(max_iter=100), TokenLogisticTagger(max_iter=100)):
            with self.subTest(model=model.name):
                model.fit(tokens, tags)
                predictions = model.predict([tokens[0], [], tokens[1]])
                self.assertEqual([len(sequence) for sequence in predictions], [3, 0, 3])
                for sequence in predictions:
                    if sequence:
                        validate_training_data([tokens[0]], [sequence])

    def test_load_bio_validates_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bio_en_entities_train.json"
            path.write_text(
                json.dumps([{"tokens": [["apt29", 0, 5, "B-ACTOR"]]}, {"tokens": []}]),
                encoding="utf-8",
            )
            tokens, tags = load_bio("entities", "train", root=directory)
        self.assertEqual(tokens, [["apt29"]])
        self.assertEqual(tags, [["B-ACTOR"]])


class PreprocessingTests(unittest.TestCase):
    def test_cleaning_and_defanging_preserve_original_before_defang(self):
        document = TextCleaner().process(Document("  APT29\u200b  hxxps://x[.]com "))
        before_defang = document.text
        Defanger().process(document)
        self.assertEqual(document.text, "APT29 https://x.com")
        self.assertEqual(document.metadata["text_before_defang"], before_defang)

    def test_language_detection_handles_punctuation_and_unicode_letters(self):
        detector = LanguageDetector()
        self.assertEqual(
            detector.process(Document("हा हल्ला खूप धोकादायक आहे!")).metadata["language"],
            "mr",
        )
        self.assertEqual(
            detector.process(Document("ગુજરાતી")).metadata["language"],
            "gu",
        )

    def test_splitter_handles_abbreviations_and_keeps_offsets(self):
        text = "APT29 uses e.g. Cobalt Strike. It contacted 10.1.2.3. Next."
        sentences = SentenceSplitter().process(Document(text)).metadata["sentences"]
        self.assertEqual(
            [sentence["text"] for sentence in sentences],
            ["APT29 uses e.g. Cobalt Strike.", "It contacted 10.1.2.3.", "Next."],
        )
        for sentence in sentences:
            self.assertEqual(
                text[sentence["start"]:sentence["end"]],
                sentence["text"],
            )

    def test_tokenizer_offsets_and_bio_span_alignment(self):
        text = "APT29 used Mimikatz"
        tokenized = tokenize(text)
        self.assertEqual(
            [text[start:end] for _, start, end in tokenized],
            [token for token, _, _ in tokenized],
        )
        bio = to_bio(
            text,
            [{"label": "ThreatActor", "start": 0, "end": 5}],
            {"ThreatActor": "THREAT_ACTOR"},
        )
        self.assertEqual(bio[0][3], "B-THREAT_ACTOR")


class SplitLockTests(unittest.TestCase):
    def setUp(self):
        self.orig_env = os.environ.get("CTI_ALLOW_TEST")
        if "CTI_ALLOW_TEST" in os.environ:
            del os.environ["CTI_ALLOW_TEST"]

        # Exercise both sides of the lock without opening held-out gold data.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.raw_dir = root / "raw"
        self.splits_dir = root / "splits"
        self.bio_dir = root / "processed"
        for folder in (self.raw_dir, self.splits_dir, self.bio_dir):
            folder.mkdir()
        records = []
        for split in ("train", "dev", "test"):
            doc_id = f"synthetic-{split}"
            records.append({"uid": doc_id, "doc_id": doc_id})
            (self.splits_dir / f"{split}_report_ids.json").write_text(
                json.dumps([doc_id]), encoding="utf-8"
            )
        (self.raw_dir / "SMAH-CTI.json").write_text(
            json.dumps(records), encoding="utf-8"
        )
        (self.bio_dir / "bio_en_entities_test.json").write_text(
            json.dumps([{"tokens": [["APT29", 0, 5, "B-THREAT_ACTOR"]]}]),
            encoding="utf-8",
        )

    def load_synthetic_smah(self, **kwargs):
        return load_smah(
            raw_dir=self.raw_dir, splits_dir=self.splits_dir, **kwargs
        )

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["CTI_ALLOW_TEST"] = self.orig_env
        elif "CTI_ALLOW_TEST" in os.environ:
            del os.environ["CTI_ALLOW_TEST"]

    def test_load_bio_refuses_test_without_env_var(self):
        with self.assertRaises(PermissionError):
            load_bio("entities", "test", root=self.bio_dir)

    def test_load_bio_allows_test_with_env_var(self):
        os.environ["CTI_ALLOW_TEST"] = "1"
        tokens, tags = load_bio("entities", "test", root=self.bio_dir)
        self.assertEqual(tokens, [["APT29"]])
        self.assertEqual(tags, [["B-THREAT_ACTOR"]])

    def test_load_smah_refuses_test_split_without_env_var(self):
        with self.assertRaises(PermissionError):
            self.load_synthetic_smah(splits=["test"])

    def test_load_smah_default_omits_test_without_env_var(self):
        parts = self.load_synthetic_smah()
        self.assertIn("train", parts)
        self.assertIn("dev", parts)
        self.assertNotIn("test", parts)

    def test_load_smah_allows_test_with_env_var(self):
        os.environ["CTI_ALLOW_TEST"] = "1"
        parts = self.load_synthetic_smah()
        self.assertIn("test", parts)
        self.assertEqual(parts["test"]["doc_id"].tolist(), ["synthetic-test"])


if __name__ == "__main__":
    unittest.main()
