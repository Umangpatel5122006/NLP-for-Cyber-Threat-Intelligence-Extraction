"""Encoding tests and isolated real-CRF plumbing with explicitly toy features.

Toy feature overrides exercise labels/serialization without the tokenizer stack.
They do not validate the production feature matrix or measure project accuracy.
"""

import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cti.models.action_bilou import INTERNAL_TAGS, bilou_is_valid, bilou_to_bio, bio_to_bilou
from cti.models.action_bilou_crf import ActionBILOUCRFTagger, load_candidate, save_candidate
from cti.models.base import validate_training_data
from cti.models.ner_artifact import load_ner_tagger


class BILOUConversionTests(unittest.TestCase):
    def test_gold_singletons_multitoken_and_adjacent_spans_have_explicit_ends(self):
        gold = ["B-ACTION", "B-ACTION", "I-ACTION", "O", "B-ACTION", "I-ACTION", "I-ACTION"]
        encoded = bio_to_bilou(gold)
        self.assertEqual(encoded, ["U-ACTION", "B-ACTION", "L-ACTION", "O", "B-ACTION", "I-ACTION", "L-ACTION"])
        self.assertEqual(bilou_to_bio(encoded), gold)

    def test_gold_rejects_orphan_inside_and_other_layer_labels(self):
        for tags in (["I-ACTION"], ["B-MALWARE_TOOL"], ["L-ACTION"], ["B-ACTION", "O", "I-ACTION"]):
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                bio_to_bilou(tags)

    def test_prediction_normalizer_preserves_explicit_closes(self):
        self.assertEqual(bilou_to_bio(["U-ACTION", "L-ACTION", "I-ACTION", "O"]),
                         ["B-ACTION", "B-ACTION", "B-ACTION", "O"])
        self.assertEqual(bilou_to_bio(["B-ACTION", "I-ACTION", "O"]), ["B-ACTION", "I-ACTION", "O"])

    def test_internal_validity_distinguishes_unclosed_and_orphan_sequences(self):
        for tags in ([], ["O"], ["U-ACTION"], ["B-ACTION", "L-ACTION"], ["B-ACTION", "I-ACTION", "L-ACTION"]):
            self.assertTrue(bilou_is_valid(tags))
        for tags in (["B-ACTION"], ["L-ACTION"], ["I-ACTION"], ["B-ACTION", "O"], ["bad"]):
            self.assertFalse(bilou_is_valid(tags))

    def test_normalization_rejects_unknown_internal_labels(self):
        with self.assertRaises(ValueError):
            bilou_to_bio(["U-MALWARE_TOOL"])

    def test_conversion_does_not_mutate_gold_or_internal_predictions(self):
        tags = ["B-ACTION", "I-ACTION", "O", "B-ACTION"]
        original = copy.deepcopy(tags)
        internal = bio_to_bilou(tags)
        saved = internal[:]
        bilou_to_bio(internal)
        self.assertEqual(tags, original)
        self.assertEqual(internal, saved)

    def test_exhaustive_legal_bio_roundtrip_preserves_every_boundary(self):
        count = 0
        for sequence in itertools.product(("O", "B-ACTION", "I-ACTION"), repeat=6):
            tags = list(sequence)
            try:
                validate_training_data([[""] * len(tags)], [tags])
            except ValueError:
                continue
            internal = bio_to_bilou(tags)
            self.assertTrue(bilou_is_valid(internal))
            self.assertEqual(bilou_to_bio(internal), tags)
            count += 1
        self.assertGreater(count, 100)

    def test_exhaustive_internal_paths_normalize_to_valid_bio(self):
        for sequence in itertools.product(sorted(INTERNAL_TAGS), repeat=4):
            internal = list(sequence)
            bio = bilou_to_bio(internal)
            validate_training_data([[""] * 4], [bio])
            if bilou_is_valid(internal):
                self.assertEqual(bio_to_bilou(bio), internal)


def toy_features(tokens, config, base, *, tags=None, lexicon=None):
    return [[{"word": word, "position": i} for i, word in enumerate(words)] for words in tokens], {}


class BILOUToyCRFTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch("cti.models.action_bilou_crf._feature_matrix", side_effect=toy_features)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.tokens = [["steal", "data"], ["access"], ["plain", "text"]] * 4
        self.tags = [["B-ACTION", "I-ACTION"], ["B-ACTION"], ["O", "O"]] * 4

    def fitted(self):
        return ActionBILOUCRFTagger(c1=0, c2=.01, max_iterations=20).fit(self.tokens, self.tags)

    def test_actual_crf_trains_internal_end_and_unit_states_and_returns_bio(self):
        model = self.fitted()
        self.assertTrue({"L-ACTION", "U-ACTION"} <= set(model.model.classes_))
        predicted = model.predict(self.tokens)
        validate_training_data(self.tokens, predicted)
        self.assertEqual(predicted, [bilou_to_bio(row) for row in model.predict_internal(self.tokens)])

    def test_empty_positions_and_dictionary_remain_unchanged_during_prediction(self):
        model = self.fitted()
        lexicon = copy.deepcopy(model.lexicon_)
        result = model.predict([[], ["new", "text"], []])
        self.assertEqual([len(row) for row in result], [0, 2, 0])
        self.assertEqual(model.lexicon_, lexicon)

    def test_explicit_artifact_roundtrip_and_default_loader_rejection(self):
        model = self.fitted()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidate.joblib"
            manifest = save_candidate(model, path, {"synthetic_toy_features": True})
            restored, saved = load_candidate(path)
            self.assertEqual(manifest, saved)
            self.assertEqual(model.predict(self.tokens), restored.predict(self.tokens))
            self.assertEqual(saved["internal_schema"], "BILOU")
            self.assertEqual(saved["output_schema"], "BIO")
            with self.assertRaises(ValueError):
                load_ner_tagger(path, expected_layer="action")

    def test_artifact_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidate.joblib"
            save_candidate(self.fitted(), path, {"synthetic_toy_features": True})
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_candidate(path)

    def test_source_drift_rejects_prediction(self):
        model = self.fitted()
        with patch("cti.models.action_bilou_crf.sha256_file", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "source changed"):
                model.predict(self.tokens)

    def test_unfitted_bad_labels_and_dev_inputs_are_rejected(self):
        model = ActionBILOUCRFTagger()
        with self.assertRaises(RuntimeError):
            model.predict([["word"]])
        for tokens, tags in (([[]], [[]]), ([["word"]], [["B-MALWARE_TOOL"]]), ([["word"]], [["I-ACTION"]])):
            with self.assertRaises(ValueError):
                model.fit(tokens, tags)
        with self.assertRaises(ValueError):
            model.fit(self.tokens, self.tags, dev_tokens=self.tokens, dev_tags=self.tags)


if __name__ == "__main__":
    unittest.main()
