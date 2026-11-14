"""Saved TF-IDF sentence relevance classification and document annotation."""

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.utils.validation import check_is_fitted

from cti.core.base import BaseModule, Document
from cti.core.registry import register
from cti.preprocessing.cleaner import TextCleaner
from cti.preprocessing.defanger import Defanger


ARTIFACT_VERSION = 1
MODEL_KIND = "tfidf_logreg_sentence_relevance"
NORMALIZATION = "cleaner_defanger_v1"


def _threshold(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("threshold must be finite and between 0 and 1.")
    return value


def _texts(texts):
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of sentence strings.")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Every sentence must be a string.")
    return list(texts)


def _normalized_texts(texts):
    cleaner, defanger = TextCleaner(), Defanger()
    return [defanger.process(cleaner.process(Document(text))).text for text in _texts(texts)]


class SentenceRelevanceClassifier:
    """Word unigram/bigram TF-IDF + balanced LogReg with shared normalization."""

    def __init__(self, threshold=0.5, c=1.0, min_df=2, max_iter=1000, seed=1):
        self.threshold = _threshold(threshold)
        if not math.isfinite(c) or c <= 0:
            raise ValueError("c must be finite and positive.")
        if not isinstance(min_df, int) or min_df < 1:
            raise ValueError("min_df must be a positive integer.")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        self.c, self.min_df, self.max_iter, self.seed = c, min_df, max_iter, seed
        self.pipeline = None
        self.metadata = {}
        self.artifact_sha256 = None

    @property
    def parameters(self):
        return {
            "threshold": self.threshold,
            "c": self.c,
            "min_df": self.min_df,
            "max_iter": self.max_iter,
            "seed": self.seed,
        }

    def fit(self, texts, labels):
        texts = _normalized_texts(texts)
        if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
            raise TypeError("labels must be a sequence of binary integers or booleans.")
        if len(texts) != len(labels) or not texts:
            raise ValueError("Training texts and labels must be nonempty and aligned.")
        if any(not isinstance(label, (int, bool)) or label not in (0, 1) for label in labels):
            raise ValueError("Labels must be binary integers or booleans.")
        labels = [int(label) for label in labels]
        if set(labels) != {0, 1}:
            raise ValueError("Training requires both relevant and irrelevant examples.")
        pipeline = SklearnPipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, sublinear_tf=True)),
            ("logreg", LogisticRegression(
                C=self.c, class_weight="balanced", max_iter=self.max_iter,
                random_state=self.seed,
            )),
        ])
        pipeline.fit(texts, labels)
        self.pipeline = pipeline
        self.metadata = {}
        self.artifact_sha256 = None
        return self

    def predict_scores(self, texts):
        if self.pipeline is None:
            raise RuntimeError("The relevance classifier must be fitted or loaded first.")
        texts = _normalized_texts(texts)
        if not texts:
            return []
        # fit/load enforce classes [0, 1]; column 1 is sentence relevance.
        return [float(score) for score in self.pipeline.predict_proba(texts)[:, 1]]

    def predict(self, texts, threshold=None):
        cutoff = self.threshold if threshold is None else _threshold(threshold)
        return [score >= cutoff for score in self.predict_scores(texts)]

    def save(self, path, metadata=None):
        if self.pipeline is None:
            raise RuntimeError("Cannot save an unfitted relevance classifier.")
        path = Path(path)
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "model_kind": MODEL_KIND,
            "normalization": NORMALIZATION,
            "parameters": self.parameters,
            "metadata": self.metadata if metadata is None else metadata,
        }
        # Validate JSON metadata before writing either artifact file.
        json.dumps(manifest, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "artifact_version": ARTIFACT_VERSION,
            "model_kind": MODEL_KIND,
            "normalization": NORMALIZATION,
            "parameters": self.parameters,
            "pipeline": self.pipeline,
        }, path, compress=3)
        manifest["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        self.metadata = manifest["metadata"]
        self.artifact_sha256 = manifest["artifact_sha256"]
        return manifest

    @classmethod
    def load(cls, path):
        """Load a trusted local joblib model and its matching manifest."""
        path = Path(path)
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("artifact_version") != ARTIFACT_VERSION
            or manifest.get("model_kind") != MODEL_KIND
            or manifest.get("normalization") != NORMALIZATION
        ):
            raise ValueError("Unsupported relevance model artifact version or kind.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("artifact_sha256"):
            raise ValueError("Relevance artifact hash does not match its manifest.")
        payload = joblib.load(path)
        if not isinstance(payload, dict) or any(
            payload.get(key) != manifest.get(key)
            for key in ("artifact_version", "model_kind", "normalization", "parameters")
        ):
            raise ValueError("Relevance model payload does not match its manifest.")
        pipeline = payload.get("pipeline")
        if not isinstance(pipeline, SklearnPipeline):
            raise ValueError("Relevance artifact has no sklearn pipeline.")
        check_is_fitted(pipeline)
        if list(pipeline.classes_) != [0, 1]:
            raise ValueError("Relevance model must predict binary classes [0, 1].")
        model = cls(**payload["parameters"])
        model.pipeline = pipeline
        model.metadata = manifest.get("metadata", {})
        model.artifact_sha256 = manifest["artifact_sha256"]
        return model


@register("relevance_filter")
class RelevanceFilter(BaseModule):
    """Annotate sentences for downstream gating while retaining text and IOCs."""

    name = "relevance_filter"

    def __init__(self, model_path, threshold=None):
        self.model = SentenceRelevanceClassifier.load(model_path)
        self.threshold = self.model.threshold if threshold is None else _threshold(threshold)

    def process(self, doc: Document) -> Document:
        if "language" not in doc.metadata or "sentences" not in doc.metadata:
            raise ValueError("Run language_detector and sentence_splitter before relevance_filter.")
        sentences = doc.metadata["sentences"]
        if not isinstance(sentences, list) or any(not isinstance(s, dict) for s in sentences):
            raise ValueError("sentences must be a list of sentence mappings.")
        texts = _texts([sentence.get("text") for sentence in sentences])
        language = doc.metadata["language"]
        supported = language == "en"
        scores = self.model.predict_scores(texts) if supported else [None] * len(texts)
        for sentence, score in zip(sentences, scores):
            sentence["relevance_score"] = score
            sentence["relevant"] = score >= self.threshold if score is not None else None
        doc.metadata["relevance"] = {
            "status": "ok" if supported else "unsupported_language",
            "language": language,
            "model_kind": MODEL_KIND,
            "artifact_version": ARTIFACT_VERSION,
            "artifact_sha256": self.model.artifact_sha256,
            "normalization": NORMALIZATION,
            "threshold": self.threshold,
            "sentence_count": len(sentences),
            "relevant_sentence_count": sum(s["relevant"] is True for s in sentences) if supported else None,
        }
        return doc
