"""Experimental classical action CRF with internal BILOU and external BIO.

The existing feature profile and annotations are retained. Internal unconstrained
decodes use a fixed, disclosed normalizer; no confidence threshold is selected.
Experimental artifacts have a separate explicit format, never the default NER
format, so internal L/U labels are not hidden from the ordinary artifact checks.
"""

import hashlib
import json
from pathlib import Path

import joblib
import sklearn_crfsuite

from cti.models.action_bilou import INTERNAL_TAGS, bilou_to_bio, bio_to_bilou
from cti.models.base import BaseTagger, validate_token_sequences, validate_training_data
from cti.models.ner_artifact import _environment, sha256_file


FORMAT_VERSION = 1
EXTRA_SOURCES = ("src/cti/models/action_crf.py", "src/cti/models/action_bilou_crf.py", "src/cti/models/action_bilou.py")
PROFILE_FLAGS = ("use_action_morphology", "use_action_lexicon")


def _feature_matrix(tokens, config, base_config, *, tags=None, lexicon=None):
    # Keep representation/serialization usable without importing the tokenizer.
    # Actual experiment fits always use this unchanged classical feature stack.
    from cti.models.action_crf import action_features, boundary_counts
    from cti.models.crf import _sentence_features

    if tags is not None:
        lexicon = boundary_counts(tokens, tags) if config.get("use_action_lexicon", False) else None
    matrix = [action_features(sentence, _sentence_features(sentence, **base_config),
              config.get("use_action_morphology", False), lexicon,
              boundary_counts([sentence], [tags[i]]) if tags is not None and lexicon is not None else None)
              for i, sentence in enumerate(tokens)]
    return matrix, lexicon


class ActionBILOUCRFTagger(BaseTagger):
    name = "crf_tagger"

    def __init__(self, algorithm="lbfgs", c1=1.0, c2=1.0, max_iterations=200,
                 min_freq=0, all_possible_transitions=True, feature_config=None):
        if algorithm != "lbfgs" or c1 < 0 or c2 < 0 or max_iterations < 1:
            raise ValueError("Use the fixed classical lbfgs CRF configuration.")
        self.parameters = {"algorithm": algorithm, "c1": c1, "c2": c2, "max_iterations": max_iterations,
                           "min_freq": min_freq, "all_possible_transitions": all_possible_transitions,
                           "feature_config": dict(feature_config or {})}
        if any(type(self.parameters["feature_config"].get(flag, False)) is not bool for flag in PROFILE_FLAGS):
            raise ValueError("Action feature flags must be boolean.")
        self.base_feature_config = {k: v for k, v in self.parameters["feature_config"].items() if k not in PROFILE_FLAGS}
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2, max_iterations=max_iterations,
                                         min_freq=min_freq, all_possible_transitions=all_possible_transitions)
        self.lexicon_, self._is_fitted, self.source_sha256_ = {}, False, None

    def fit(self, tokens, tags, dev_tokens=None, dev_tags=None):
        validate_training_data(tokens, tags)
        if dev_tokens is not None or dev_tags is not None:
            raise ValueError("Fit takes training inputs only.")
        # Encode before filtering empties so every label sequence is validated.
        encoded = [bio_to_bilou(sequence) for sequence in tags]
        rows = [(words, gold, internal) for words, gold, internal in zip(tokens, tags, encoded) if words]
        if not rows:
            raise ValueError("Cannot fit empty action training data.")
        words, gold, internal = map(list, zip(*rows))
        config = self.parameters["feature_config"]
        matrix, lexicon = _feature_matrix(words, config, self.base_feature_config, tags=gold)
        source_sha = sha256_file(Path(__file__))
        self.model.fit(matrix, internal)
        self.lexicon_, self._is_fitted = lexicon or {}, True
        self.source_sha256_ = source_sha
        return self

    def predict_internal(self, tokens):
        validate_token_sequences(tokens)
        if not self._is_fitted:
            raise RuntimeError("The BILOU action CRF must be fitted first.")
        if self.source_sha256_ != sha256_file(Path(__file__)):
            raise ValueError("BILOU source changed; retrain the experimental candidate.")
        selected, output = [i for i, words in enumerate(tokens) if words], [[] for _ in tokens]
        if selected:
            config = self.parameters["feature_config"]
            lexicon = self.lexicon_ if config.get("use_action_lexicon", False) else None
            matrix, _ = _feature_matrix([tokens[i] for i in selected], config, self.base_feature_config, lexicon=lexicon)
            decoded = list(self.model.predict(matrix))
            if len(decoded) != len(selected):
                raise ValueError("Internal prediction sentence counts differ.")
            for i, row in zip(selected, decoded):
                if len(row) != len(tokens[i]) or any(tag not in INTERNAL_TAGS for tag in row):
                    raise ValueError("Internal prediction tags/tokens differ.")
                output[i] = list(row)
        return output

    def predict(self, tokens):
        return [bilou_to_bio(row) for row in self.predict_internal(tokens)]


def save_candidate(model, path, metadata):
    if not isinstance(model, ActionBILOUCRFTagger) or model.predict([[]]) != [[]]:
        raise ValueError("Expected a fitted experimental BILOU action model.")
    manifest = {"version": FORMAT_VERSION, "kind": "classical_action_bilou_experiment", "model_name": "crf_tagger",
                "layer": "action", "language": "en", "internal_schema": "BILOU", "output_schema": "BIO",
                "normalization": "open_span_v1", "parameters": model.parameters,
                "environment": _environment(EXTRA_SOURCES), "metadata": metadata}
    json.dumps(manifest, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"manifest": manifest, "tagger": model}, path, compress=3)
    result = {**manifest, "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def load_candidate(path):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    fixed = {"version": FORMAT_VERSION, "kind": "classical_action_bilou_experiment", "model_name": "crf_tagger", "layer": "action",
             "language": "en", "internal_schema": "BILOU", "output_schema": "BIO", "normalization": "open_span_v1"}
    if any(manifest.get(k) != v for k, v in fixed.items()) or manifest.get("environment") != _environment(EXTRA_SOURCES):
        raise ValueError("BILOU artifact schema, resources or runtime changed.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("artifact_sha256"):
        raise ValueError("BILOU artifact hash differs from its manifest.")
    payload = joblib.load(path)
    expected = {k: v for k, v in manifest.items() if k != "artifact_sha256"}
    model = payload.get("tagger") if isinstance(payload, dict) else None
    if (not isinstance(model, ActionBILOUCRFTagger) or payload.get("manifest") != expected
            or model.parameters != manifest["parameters"] or model.predict([[]]) != [[]]
            or any(tag not in INTERNAL_TAGS for tag in model.model.classes_)):
        raise ValueError("BILOU artifact payload/model differs from the declared profile.")
    return model, manifest
