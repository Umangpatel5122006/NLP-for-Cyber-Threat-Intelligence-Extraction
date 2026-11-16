"""Production-stack checks; require the compatible classical NLP runtime."""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cti.experiment_action import _digest
from cti.experiment_action_bilou import decode_cache, read_config
from cti.models.action_bilou_crf import ActionBILOUCRFTagger


ROOT = Path(__file__).resolve().parents[1]


class BILOUProductionTests(unittest.TestCase):
    def test_production_features_fit_and_preserve_prediction_alignment(self):
        model = ActionBILOUCRFTagger(c1=0, c2=.01, max_iterations=15,
                feature_config={"use_action_morphology": True, "use_action_lexicon": True})
        tokens = [["steal", "data"], ["plain", "text"]] * 3
        tags = [["B-ACTION", "I-ACTION"], ["O", "O"]] * 3
        model.fit(tokens, tags)
        self.assertEqual([len(row) for row in model.predict([[], tokens[0]])], [0, 2])

    def test_cache_checks_internal_hash_and_reproduces_fixed_normalization(self):
        internal, predicted = [["U-ACTION", "O"]], [["B-ACTION", "O"]]
        cache = {"signature": {"synthetic": True}, "internal": internal, "internal_sha256": _digest(internal),
                 "predictions": predicted, "predictions_sha256": _digest(predicted)}
        self.assertEqual(decode_cache(cache, {"synthetic": True}, [["word", "text"]]), (internal, predicted))
        for changed in ({**cache, "internal_sha256": "bad"}, {**cache, "signature": {}},
                        {**cache, "predictions": [["O", "O"]], "predictions_sha256": _digest([["O", "O"]])}):
            with self.assertRaises(ValueError):
                decode_cache(changed, {"synthetic": True}, [["word", "text"]])

    def test_configuration_rejects_nonfinite_or_boolean_budgets(self):
        config = yaml.safe_load((ROOT / "configs/action_bilou_experiment.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            for value in (float("nan"), True, -.1):
                config["selection"]["minimum_f1_gain"] = value
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_config(path)


if __name__ == "__main__":
    unittest.main()
