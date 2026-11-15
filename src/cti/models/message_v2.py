"""Versioned expanded-data SMS models; legacy artifacts remain independently loadable."""

import hashlib
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from cti.models.message_classifier import MessageClassifier, environment
from cti.models.message_features import MessageConcepts, normalize_message


PROFILES = ("normalized_svm", "concept_svm")


def new_expanded_pipeline(profile):
    if profile not in PROFILES:
        raise ValueError("Unsupported expanded classical message profile.")
    common = {"min_df": 2, "sublinear_tf": True, "preprocessor": normalize_message, "lowercase": False}
    groups = [("word", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, **common)),
              ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=60_000, **common))]
    if profile == "concept_svm":
        groups.append(("concept", MessageConcepts()))
    return Pipeline([("features", FeatureUnion(groups)), ("classifier", LinearSVC(
        C=1.0, class_weight="balanced", max_iter=10_000, random_state=42))])


def expanded_environment():
    base = environment()
    return {**base, "expanded_sources_sha256": {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in ("message_v2.py", "message_features.py")}}


class ExpandedMessageClassifier(MessageClassifier):
    @classmethod
    def save(cls, pipeline, path, profile, metadata):
        if profile not in PROFILES or list(pipeline.classes_) != ["ham", "spam"] or not isinstance(pipeline["classifier"], LinearSVC):
            raise ValueError("Expanded artifact requires a fitted supported classical classifier.")
        manifest = {"artifact_version": 2, "kind": "classical_sms_spam", "profile": profile,
                    "classes": ["ham", "spam"], "threshold": 0.0, "score_kind": "decision_margin",
                    "environment": expanded_environment(), "metadata": metadata}
        json.dumps(manifest, allow_nan=False)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "manifest": manifest}, path, compress=3)
        result = {**manifest, "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return cls(pipeline, result)

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("artifact_version") != 2 or manifest.get("kind") != "classical_sms_spam"
                or manifest.get("profile") not in PROFILES or manifest.get("classes") != ["ham", "spam"]
                or manifest.get("threshold") != 0.0 or manifest.get("score_kind") != "decision_margin"):
            raise ValueError("Unsupported expanded SMS artifact schema.")
        if manifest.get("environment") != expanded_environment():
            raise ValueError("Expanded SMS runtime/features changed; retrain in this environment.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("artifact_sha256"):
            raise ValueError("Expanded SMS artifact hash mismatch.")
        payload = joblib.load(path)
        if (not isinstance(payload, dict) or payload.get("manifest") != {k: v for k, v in manifest.items() if k != "artifact_sha256"}
                or list(payload["pipeline"].classes_) != ["ham", "spam"]
                or not isinstance(payload["pipeline"]["classifier"], LinearSVC)):
            raise ValueError("Expanded SMS payload does not match its manifest.")
        return cls(payload["pipeline"], manifest)


def load_message_classifier(path):
    manifest = json.loads(Path(path).with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_version") == 1:
        return MessageClassifier.load(path)
    if manifest.get("artifact_version") == 2:
        return ExpandedMessageClassifier.load(path)
    raise ValueError("Unsupported SMS model version.")
