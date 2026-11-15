"""Classical sparse TF-IDF SMS classification, with checked local persistence."""

import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


PROFILES = ("word_logreg", "word_char_logreg", "word_char_svm")


def new_pipeline(profile):
    if profile not in PROFILES:
        raise ValueError("Unsupported classical SMS profile.")
    groups = [("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                      strip_accents="unicode", max_features=50_000))]
    if profile != "word_logreg":
        groups.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                               sublinear_tf=True, max_features=60_000)))
    classifier = (LinearSVC(C=1.0, class_weight="balanced", max_iter=10_000, random_state=42)
                  if profile.endswith("svm") else LogisticRegression(
                      C=1.0, class_weight="balanced", solver="liblinear", max_iter=1500, random_state=42))
    return Pipeline([("features", FeatureUnion(groups)), ("classifier", classifier)])


def environment():
    return {"python": ".".join(platform.python_version_tuple()[:2]),
            "packages": {name: version(name) for name in ("scikit-learn", "numpy", "scipy", "joblib")},
            "feature_source_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


class MessageClassifier:
    def __init__(self, pipeline, manifest):
        self.pipeline, self.manifest = pipeline, manifest

    @classmethod
    def save(cls, pipeline, path, profile, metadata):
        if profile not in PROFILES or list(pipeline.classes_) != ["ham", "spam"]:
            raise ValueError("Invalid fitted SMS classifier.")
        manifest = {"artifact_version": 1, "kind": "classical_sms_spam", "profile": profile,
                    "classes": ["ham", "spam"], "threshold": 0.0, "score_kind": "decision_margin",
                    "environment": environment(), "metadata": metadata}
        json.dumps(manifest, allow_nan=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "manifest": manifest}, path, compress=3)
        sidecar = {**manifest, "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix(".manifest.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        return cls(pipeline, sidecar)

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("artifact_version") != 1 or manifest.get("kind") != "classical_sms_spam"
                or manifest.get("profile") not in PROFILES or manifest.get("classes") != ["ham", "spam"]
                or manifest.get("threshold") != 0.0 or manifest.get("score_kind") != "decision_margin"):
            raise ValueError("Unsupported SMS artifact schema.")
        if manifest.get("environment") != environment():
            raise ValueError("SMS runtime or feature source changed; retrain in the selected environment.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("artifact_sha256"):
            raise ValueError("SMS artifact hash mismatch.")
        payload = joblib.load(path)
        expected = {k: v for k, v in manifest.items() if k != "artifact_sha256"}
        if payload.get("manifest") != expected or list(payload["pipeline"].classes_) != ["ham", "spam"]:
            raise ValueError("SMS payload and manifest disagree.")
        return cls(payload["pipeline"], manifest)

    def predict_one(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("SMS text must be a nonempty string.")
        features = self.pipeline["features"].transform([text])
        classifier = self.pipeline["classifier"]
        score = float(classifier.decision_function(features)[0])
        if not math.isfinite(score):
            raise ValueError("SMS classifier returned a nonfinite margin.")
        names = self.pipeline["features"].get_feature_names_out()
        values = features.data * classifier.coef_[0, features.indices]
        order = sorted(range(len(values)), key=lambda i: float(values[i]), reverse=True)
        supporting = [{"feature": str(names[features.indices[i]]), "contribution": round(float(values[i]), 6)}
                      for i in order if values[i] > 0][:8]
        return {"status": "ok", "label": "spam" if score > 0 else "ham", "decision_margin": score,
                "threshold": 0.0, "score_kind": "decision_margin", "profile": self.manifest["profile"],
                "artifact_sha256": self.manifest["artifact_sha256"], "supporting_features": supporting,
                "task": "sms_spam_classification", "limitations": "SMS ham/spam prediction, not a verified fraud or attack verdict. Margin is not a probability."}
