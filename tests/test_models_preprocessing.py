import json
import tempfile
import unittest
from pathlib import Path

from cti.core.base import Document
from cti.core.bio import to_bio
from cti.models.base import validate_training_data
from cti.models.bert_ner import _align_word_labels
from cti.models.bilstm_crf import BiLSTMCRF
from cti.models.crf import CRFTagger
from cti.models.data import load_bio
from cti.models.memm import MEMMTagger, TokenLogisticTagger
from cti.models.spacy_ner import SpacyNER
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

    def test_bert_subword_labels_align_once_per_word(self):
        labels = _align_word_labels(
            [None, 0, 0, 1, None],
            ["B-ACTOR", "O"],
            {"B-ACTOR": 1, "O": 0},
        )
        self.assertEqual(labels, [-100, 1, -100, 0, -100])

    def test_crf_preserves_empty_sentence_positions(self):
        model = CRFTagger(max_iterations=5)
        model.fit(
            [["apt29", "used"], ["mimikatz"]],
            [["B-ACTOR", "O"], ["B-TOOL"]],
        )
        predictions = model.predict([["apt29"], [], ["mimikatz", "ran"]])
        self.assertEqual([len(sequence) for sequence in predictions], [1, 0, 2])

    def test_bilstm_crf_preserves_empty_sentence_positions(self):
        model = BiLSTMCRF(epochs=1, batch_size=2, min_freq=1)
        model.fit(
            [["apt29", "used"], ["mimikatz"]],
            [["B-ACTOR", "O"], ["B-TOOL"]],
        )
        predictions = model.predict([["apt29"], [], ["mimikatz", "ran"]])
        self.assertEqual([len(sequence) for sequence in predictions], [1, 0, 2])

    def test_spacy_ner_preserves_empty_sentence_positions(self):
        model = SpacyNER(epochs=1, dropout=0)
        model.fit(
            [["apt29", "used"], ["mimikatz"]],
            [["B-ACTOR", "O"], ["B-TOOL"]],
        )
        predictions = model.predict([["apt29"], [], ["mimikatz"]])
        self.assertEqual([len(sequence) for sequence in predictions], [1, 0, 1])

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


if __name__ == "__main__":
    unittest.main()
