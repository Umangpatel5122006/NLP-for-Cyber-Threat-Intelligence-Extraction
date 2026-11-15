import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cti.core.dataset import load_smah
from cti.models.action_crf import (
    ActionBoundaryCRFTagger, action_features, action_stem, boundary_counts, create_crf_tagger,
)
from cti.models.crf import CRFTagger, _sentence_features
from cti.models.ner_artifact import ROOT, _environment, load_ner_tagger, save_ner_tagger, sha256_file
from cti.preprocessing.tokenizer import tokenize
from cti.train_ner import train_from_config


TOKENS = [["APT29", "uses", "Mimikatz"], ["Lazarus", "uses", "malware"]] * 3
TAGS = [["O", "B-ACTION", "I-ACTION"]] * 6
PROFILE = {"use_action_morphology": True, "use_action_lexicon": True}


def fitted():
    model = create_crf_tagger(c1=0, c2=.01, max_iterations=25, feature_config=PROFILE)
    model.fit(TOKENS, TAGS)
    return model


class ActionFeatureTests(unittest.TestCase):
    def test_rule_morphology_groups_known_inflections_and_irregulars(self):
        for word, expected in [
            ("uses", "use"), ("used", "use"), ("using", "use"),
            ("executing", "execute"), ("copied", "copy"), ("dropped", "drop"),
            ("running", "run"), ("ran", "run"), ("written", "write"),
            ("sent", "send"), ("stolen", "steal"),
        ]:
            with self.subTest(word=word):
                self.assertEqual(action_stem(word), expected)

    def test_morphology_adds_cues_without_replacing_base_features(self):
        words = ["could", "not", "execute", "files", ";", "then", "stop"]
        base = _sentence_features(words)
        before = copy.deepcopy(base)
        enhanced = action_features(words, base)
        self.assertEqual(base, before)
        for old, new in zip(base, enhanced):
            self.assertTrue(old.items() <= new.items())
        self.assertEqual(enhanced[0]["action:class"], "modal")
        self.assertEqual(enhanced[1]["action:class"], "neg")
        self.assertTrue(enhanced[2]["action:verb_cue"])
        self.assertEqual(enhanced[3]["action:1:clause_distance"], 1)
        self.assertEqual(enhanced[4]["action:class"], "punct")

    def test_lexical_statistics_exclude_current_sentence_and_unseen_words(self):
        words = ["use"] * 6
        tags = ["B-ACTION"] * 6
        lexicon = boundary_counts([words], [tags])
        base = [{} for _ in words]
        training = action_features(words, base, False, lexicon, lexicon)
        prediction = action_features(words, base, False, lexicon)
        self.assertFalse(training[0]["action:lex:0:known"])
        self.assertTrue(prediction[0]["action:lex:0:known"])
        self.assertGreater(prediction[0]["action:lex:0:start"], .9)
        unseen = action_features(["devonlyword"], [{}], False, lexicon)
        self.assertFalse(unseen[0]["action:lex:0:known"])

    def test_fit_uses_no_dev_labels_and_prediction_does_not_learn_words(self):
        model = create_crf_tagger(max_iterations=10, feature_config=PROFILE)
        model.fit(TOKENS, TAGS, dev_tokens=[["devonlyword"]], dev_tags=[["B-ACTION"]])
        before = copy.deepcopy(model.lexicon_)
        self.assertNotIn("devonlyword", before)
        model.predict([["devonlyword"], [], TOKENS[0]])
        self.assertEqual(model.lexicon_, before)

    def test_empty_alignment_bio_contract_and_invalid_training_inputs(self):
        model = fitted()
        predictions = model.predict([TOKENS[0], [], TOKENS[1]])
        self.assertEqual([len(sentence) for sentence in predictions], [3, 0, 3])
        with self.assertRaisesRegex(ValueError, "action layer"):
            model.fit([["APT29"]], [["B-THREAT_ACTOR"]])
        with self.assertRaises(ValueError):
            model.fit([], [])
        with self.assertRaises(ValueError):
            model.predict(TOKENS, precomputed_X=[])
        with self.assertRaises(RuntimeError):
            create_crf_tagger(feature_config=PROFILE).predict([["use"]])

    def test_baseline_factory_and_existing_artifact_environment_stay_compatible(self):
        self.assertIs(type(create_crf_tagger()), CRFTagger)
        self.assertIs(type(create_crf_tagger(feature_config={"use_action_morphology": False})), CRFTagger)
        disabled = create_crf_tagger(max_iterations=5, feature_config={"use_action_morphology": False})
        disabled.fit(TOKENS, TAGS)
        self.assertEqual(disabled.feature_config, {})
        self.assertEqual(len(disabled.predict([TOKENS[0]])[0]), 3)
        self.assertNotIn("src/cti/models/action_crf.py", _environment()["dependency_sha256"])
        with self.assertRaises(ValueError):
            ActionBoundaryCRFTagger()
        with self.assertRaises(ValueError):
            create_crf_tagger(feature_config={"use_action_morphology": "false"})

    def test_profile_artifact_roundtrip_fingerprints_new_source_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.joblib"
            model = fitted()
            manifest = save_ner_tagger(model, path, "action")
            loaded = load_ner_tagger(path, expected_layer="action")
            self.assertEqual(loaded.tagger.predict(TOKENS), model.predict(TOKENS))
            self.assertEqual(loaded.tagger.lexicon_, model.lexicon_)
            dependencies = manifest["environment"]["dependency_sha256"]
            self.assertEqual(dependencies["src/cti/models/action_crf.py"], sha256_file(ROOT / "src/cti/models/action_crf.py"))
            real_hash = sha256_file

            def changed_source(filename):
                return "changed" if Path(filename).name == "action_crf.py" else real_hash(filename)

            with patch("cti.models.ner_artifact.sha256_file", side_effect=changed_source):
                with self.assertRaisesRegex(ValueError, "resources or runtime"):
                    load_ner_tagger(path)
            with self.assertRaises(ValueError):
                save_ner_tagger(model, path, "entities")

    def test_plain_crf_cannot_claim_an_action_feature_profile(self):
        model = CRFTagger(max_iterations=5, feature_config=PROFILE)
        # Fit via explicit base matrices to exercise the persistence guard.
        model.fit(TOKENS, TAGS, precomputed_X=[_sentence_features(words) for words in TOKENS])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "implementation"):
                save_ner_tagger(model, Path(directory) / "invalid.joblib", "action")

    def test_normal_training_cli_supports_profile_and_saved_inference(self):
        import yaml

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("raw", "splits", "processed"):
                (root / folder).mkdir()
            rows = []
            for split in ("train", "dev", "test"):
                identifier = "synthetic-" + split
                (root / f"splits/{split}_report_ids.json").write_text(json.dumps([identifier]))
                records = []
                for index, text in enumerate(["APT29 uses Mimikatz.", "Lazarus uses malware."]):
                    uid = f"{split}-{index}"
                    rows.append({"uid": uid, "doc_id": identifier, "sentence_text": text, "relevant": "yes", "source": "synthetic"})
                    pieces = tokenize(text)
                    records.append({"uid": uid, "tokens": [list(piece) + [tag] for piece, tag in zip(pieces, ["O", "B-ACTION", "I-ACTION", "O"])]})
                if split != "test":
                    (root / f"processed/bio_en_action_{split}.json").write_text(json.dumps(records))
            (root / "raw/SMAH-CTI.json").write_text(json.dumps(rows))
            config = root / "training.yaml"
            config.write_text(yaml.safe_dump({
                "train_relevant_only": True,
                "layers": {"action": {"model": "crf_tagger", "parameters": {"max_iterations": 10, "feature_config": PROFILE}}},
            }))
            reports = train_from_config(config, root / "raw", root / "splits", root / "processed", root / "models", root / "results", layers=("action",))
            self.assertTrue(reports["action"]["reload_predictions_identical"])
            model = load_ner_tagger(root / "models/ner_action.joblib", expected_layer="action").tagger
            self.assertIsInstance(model, ActionBoundaryCRFTagger)
