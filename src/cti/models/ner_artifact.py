"""Versioned local artifacts for the project's classical BIO taggers."""

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import yaml

from cti.core.model_policy import CLASSICAL_NER_MODULES, check_model_scope
from cti.models.base import BaseTagger


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_VERSION = 1
DEPENDENCY_FILES = (
    "src/cti/models/base.py", "src/cti/models/crf.py", "src/cti/models/memm.py",
    "src/cti/preprocessing/tokenizer.py", "configs/labels.yaml",
    "data/external/gazetteer.json",
)
PARAMETER_NAMES = (
    "algorithm", "c1", "c2", "max_iterations", "min_freq",
    "all_possible_transitions", "feature_config", "c", "max_iter", "seed",
)


def sha256_file(path):
    """Hash text data/code independently of Windows checkout line endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def layer_labels(layer):
    with (ROOT / "configs/labels.yaml").open(encoding="utf-8") as file:
        layers = yaml.safe_load(file)["layers"]
    if layer not in ("entities", "action"):
        raise ValueError("layer must be 'entities' or 'action'.")
    return sorted(layers[layer])


def _profile_dependencies(feature_config):
    config = feature_config or {}
    flags = ("use_action_morphology", "use_action_lexicon")
    for flag in flags:
        if type(config.get(flag, False)) is not bool:
            raise ValueError("Action feature flags must be boolean.")
    return ("src/cti/models/action_crf.py",) if any(config.get(flag, False) for flag in flags) else ()


def _environment(extra_dependencies=()):
    return {
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "packages": {name: version(name) for name in (
            "spacy", "scikit-learn", "sklearn-crfsuite", "python-crfsuite", "joblib",
        )},
        "dependency_sha256": {name: sha256_file(ROOT / name) for name in DEPENDENCY_FILES + tuple(extra_dependencies)},
    }


def _parameters(tagger):
    return {name: getattr(tagger, name) for name in PARAMETER_NAMES if hasattr(tagger, name)}


def _check_tagger(tagger, layer):
    if not isinstance(tagger, BaseTagger):
        raise TypeError("NER artifacts require a BaseTagger instance.")
    check_model_scope(tagger.name)
    if tagger.name not in CLASSICAL_NER_MODULES:
        raise ValueError("Only supported classical NER taggers can be saved.")
    if _profile_dependencies(getattr(tagger, "feature_config", None)):
        from cti.models.action_crf import ActionBoundaryCRFTagger
        if layer != "action" or not isinstance(tagger, ActionBoundaryCRFTagger):
            raise ValueError("Action feature profiles require the action-layer CRF implementation.")
    labels = layer_labels(layer)
    # The shared contract checks readiness without inventing an input token.
    if tagger.predict([[]]) != [[]]:
        raise ValueError("A fitted tagger must preserve empty sentence positions.")
    observed = getattr(getattr(tagger, "model", None), "classes_", [])
    for tag in observed:
        if tag != "O" and (tag[:2] not in ("B-", "I-") or tag[2:] not in labels):
            raise ValueError(f"Tag {tag!r} is outside the {layer!r} label layer.")
    return labels


@dataclass
class NERArtifact:
    tagger: BaseTagger
    manifest: dict

    @property
    def layer(self):
        return self.manifest["layer"]


def save_ner_tagger(tagger, path, layer, metadata=None):
    """Save a fitted tagger with the feature resources needed to reproduce it."""
    labels = _check_tagger(tagger, layer)
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "kind": "classical_bio_ner",
        "model_name": tagger.name,
        "layer": layer,
        "language": "en",
        "labels": labels,
        "parameters": _parameters(tagger),
        "environment": _environment(_profile_dependencies(getattr(tagger, "feature_config", None))),
        "metadata": {} if metadata is None else metadata,
    }
    json.dumps(manifest, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"manifest": manifest, "tagger": tagger}, path, compress=3)
    result = {**manifest, "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def load_ner_tagger(path, expected_layer=None):
    """Load a trusted locally trained joblib artifact; reject feature drift."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != ARTIFACT_VERSION or manifest.get("kind") != "classical_bio_ner":
        raise ValueError("Unsupported NER artifact version or kind.")
    check_model_scope(manifest.get("model_name"))
    if manifest.get("model_name") not in CLASSICAL_NER_MODULES:
        raise ValueError("Unsupported classical NER model.")
    if expected_layer is not None and manifest.get("layer") != expected_layer:
        raise ValueError(f"Expected NER layer {expected_layer!r}, got {manifest.get('layer')!r}.")
    if manifest.get("language") != "en" or manifest.get("labels") != layer_labels(manifest.get("layer")):
        raise ValueError("NER artifact language or label schema does not match this project.")
    profile = manifest.get("parameters", {}).get("feature_config")
    if manifest.get("environment") != _environment(_profile_dependencies(profile)):
        raise ValueError("NER feature resources or runtime versions changed; retrain in this environment.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("artifact_sha256"):
        raise ValueError("NER artifact hash does not match its manifest.")
    payload = joblib.load(path)
    expected = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if not isinstance(payload, dict) or payload.get("manifest") != expected:
        raise ValueError("NER artifact payload and manifest disagree.")
    tagger = payload.get("tagger")
    _check_tagger(tagger, manifest["layer"])
    if tagger.name != manifest["model_name"] or _parameters(tagger) != manifest["parameters"]:
        raise ValueError("NER tagger does not match its declared model/parameters.")
    return NERArtifact(tagger, manifest)
