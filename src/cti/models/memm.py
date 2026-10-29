from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from cti.core.registry import register
from cti.models.base import (
    BaseTagger,
    repair_bio_sequence,
    validate_token_sequences,
    validate_training_data,
)
from cti.models.crf import _sentence_features


class _TokenClassifier(BaseTagger):
    """Shared sklearn token classifier with optional previous-tag history."""

    use_previous_tag = False

    def __init__(self, c=1.0, max_iter=1000, seed=1):
        if c <= 0 or max_iter < 1:
            raise ValueError("c and max_iter must be positive.")
        self.c = c
        self.max_iter = max_iter
        self.seed = seed
        self.vectorizer = None
        self.model = None

    def _training_rows(self, tokens, tags):
        rows, labels = [], []
        for sentence, sequence_tags in zip(tokens, tags):
            sentence_features = _sentence_features(sentence)
            previous_tag = "BOS"
            for features, tag in zip(sentence_features, sequence_tags):
                row = features.copy()
                if self.use_previous_tag:
                    row["previous_tag"] = previous_tag
                rows.append(row)
                labels.append(tag)
                previous_tag = tag
        return rows, labels

    def fit(self, tokens, tags, dev_tokens=None, dev_tags=None):
        validate_training_data(tokens, tags)
        if (dev_tokens is None) != (dev_tags is None):
            raise ValueError("dev_tokens and dev_tags must be provided together.")
        if dev_tokens is not None:
            validate_training_data(dev_tokens, dev_tags)

        nonempty = [
            (sentence, sequence_tags)
            for sentence, sequence_tags in zip(tokens, tags)
            if sentence
        ]
        if not nonempty:
            raise ValueError(f"Cannot fit {self.name} on an empty training set.")
        train_tokens, train_tags = map(list, zip(*nonempty))
        rows, labels = self._training_rows(train_tokens, train_tags)
        self.vectorizer = DictVectorizer(sparse=True)
        matrix = self.vectorizer.fit_transform(rows)
        self.model = LogisticRegression(
            C=self.c,
            max_iter=self.max_iter,
            random_state=self.seed,
            tol=1e-3,
        )
        self.model.fit(matrix, labels)

    def _predict_independent(self, feature_sequences):
        rows = [features for sequence in feature_sequences for features in sequence]
        if not rows:
            return [[] for _ in feature_sequences]
        predictions = self.model.predict(self.vectorizer.transform(rows))
        result, offset = [], 0
        for sequence in feature_sequences:
            result.append(list(predictions[offset:offset + len(sequence)]))
            offset += len(sequence)
        return result

    def _predict_with_history(self, feature_sequences):
        result = []
        for sequence in feature_sequences:
            predicted = []
            previous_tag = "BOS"
            for features in sequence:
                row = features.copy()
                row["previous_tag"] = previous_tag
                probabilities = self.model.predict_proba(
                    self.vectorizer.transform([row])
                )[0]
                ranked = probabilities.argsort()[::-1]
                tag = None
                for class_index in ranked:
                    candidate = self.model.classes_[class_index]
                    if candidate.startswith("I-"):
                        label = candidate[2:]
                        if previous_tag not in (f"B-{label}", f"I-{label}"):
                            continue
                    tag = candidate
                    break
                if tag is None:
                    raise RuntimeError("MEMM could not find a valid BIO transition.")
                predicted.append(tag)
                previous_tag = tag
            result.append(repair_bio_sequence(predicted))
        return result

    def predict(self, tokens):
        validate_token_sequences(tokens)
        if self.model is None:
            raise RuntimeError(f"{self.name} must be fitted before calling predict().")

        feature_sequences = [
            _sentence_features(sentence) if sentence else [] for sentence in tokens
        ]
        if self.use_previous_tag:
            return self._predict_with_history(feature_sequences)
        return [
            repair_bio_sequence(sequence)
            for sequence in self._predict_independent(feature_sequences)
        ]


@register("token_logreg")
class TokenLogisticTagger(_TokenClassifier):
    """Independent per-token multinomial logistic-regression baseline."""

    name = "token_logreg"


@register("memm_tagger")
class MEMMTagger(_TokenClassifier):
    """Maximum Entropy Markov Model using previous predicted BIO tags."""

    name = "memm_tagger"
    use_previous_tag = True
