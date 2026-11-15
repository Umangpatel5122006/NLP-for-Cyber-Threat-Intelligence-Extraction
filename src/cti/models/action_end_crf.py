"""Isolated classical action-ending profiles; existing CRF code is unchanged.

Fixed token cues are not POS/dependency predictions. Optional stem-pair boundary
statistics use fit inputs only, excluding the current sentence during fitting.
"""

from collections import defaultdict
from pathlib import Path

from cti.models.action_crf import (ActionBoundaryCRFTagger, _category, action_features,
                                   action_stem, boundary_counts)
from cti.models.base import validate_token_sequences, validate_training_data
from cti.models.crf import CRFTagger, _sentence_features
from cti.models.ner_artifact import load_ner_tagger, save_ner_tagger, sha256_file


CLAUSE_WORDS = frozenset("that which where when while whereas because although before after until unless".split())
PROFILE_VERSION = 1


def edge_counts(tokens, tags):
    counts = defaultdict(lambda: [0, 0, 0, 0])
    for words, labels in zip(tokens, tags):
        for i in range(len(words)-1):
            key = (action_stem(words[i]), action_stem(words[i+1]))
            values = counts[key]
            inside = labels[i] != "O"
            values[0] += 1
            values[1] += inside
            values[2] += inside and labels[i+1] != "I-ACTION"
            values[3] += labels[i+1] == "B-ACTION"
    return {key: tuple(values) for key, values in counts.items()}


def ending_features(words, base, edges=None, exclude=None):
    features = [dict(row) for row in base]
    categories, stems = [_category(w) for w in words], [action_stem(w) for w in words]
    for i, values in enumerate(features):
        following = words[i+1].casefold() if i+1 < len(words) else "<EOS>"
        next_class = categories[i+1] if i+1 < len(words) else "EOS"
        third_class = categories[i+2] if i+2 < len(words) else "EOS"
        values.update({"ending:next_clause": following in CLAUSE_WORDS,
                       "ending:next_punct": following in (".", ",", ";", ":", "!", "?"),
                       "ending:next_close": following in (")", "]", "}", '"', "\u201d"),
                       "ending:next_coord_verb": following in ("and", "or", "but") and third_class in ("verb_cue", "aux", "modal"),
                       "ending:classes": categories[i] + "|" + next_class + "|" + third_class,
                       "ending:nominal_suffix": words[i].casefold().endswith(("tion", "ment", "ance", "ence", "ity", "ness"))})
        if edges is not None:
            for offset, side in ((0, "right"), (-1, "left")):
                j = i + offset
                if not 0 <= j < len(words)-1:
                    continue
                key = (stems[j], stems[j+1])
                totals = edges.get(key, (0, 0, 0, 0))
                removed = (exclude or {}).get(key, (0, 0, 0, 0))
                total, inside, ends, next_starts = [a-b for a, b in zip(totals, removed)]
                if min(total, inside, ends, next_starts) < 0 or ends > inside or inside > total or next_starts > total:
                    raise ValueError("Boundary edge exclusion exceeds fitted counts.")
                prefix = "ending:edge:" + side + ":"
                values[prefix + "known"] = total >= 3
                if total >= 3:
                    values[prefix + "end_rate"] = (ends + .5) / (total + 1)
                    values[prefix + "end_given_inside"] = (ends + .5) / (inside + 1)
                    values[prefix + "next_start_rate"] = (next_starts + .5) / (total + 1)
    return features


class ActionEndCRFTagger(ActionBoundaryCRFTagger):
    """Experimental subclass with unchanged CRF family and action BIO labels."""

    def __init__(self, learned_edges=False, **parameters):
        if type(learned_edges) is not bool:
            raise ValueError("learned_edges must be boolean.")
        super().__init__(**parameters)
        self.learned_edges = learned_edges
        self.edge_lexicon_ = {}
        self.profile_source_sha256_ = None

    def fit(self, tokens, tags, dev_tokens=None, dev_tags=None, precomputed_X=None):
        validate_training_data(tokens, tags)
        if dev_tokens is not None or dev_tags is not None or precomputed_X is not None:
            raise ValueError("Use fit-only inputs; this profile builds its own matrices.")
        if any(tag not in ("O", "B-ACTION", "I-ACTION") for row in tags for tag in row):
            raise ValueError("Action-ending profiles accept action BIO labels only.")
        pairs = [(words, labels) for words, labels in zip(tokens, tags) if words]
        if not pairs:
            raise ValueError("Cannot fit empty action training data.")
        words, labels = map(list, zip(*pairs))
        lexicon = boundary_counts(words, labels) if self.feature_config.get("use_action_lexicon", False) else None
        edges = edge_counts(words, labels) if self.learned_edges else None
        matrix = []
        for sentence, sequence in pairs:
            base = action_features(sentence, _sentence_features(sentence, **self.base_feature_config),
                self.feature_config.get("use_action_morphology", False), lexicon,
                boundary_counts([sentence], [sequence]) if lexicon is not None else None)
            matrix.append(ending_features(sentence, base, edges, edge_counts([sentence], [sequence]) if edges is not None else None))
        self.profile_source_sha256_ = sha256_file(Path(__file__))
        CRFTagger.fit(self, words, labels, precomputed_X=matrix)
        self.lexicon_, self.edge_lexicon_ = lexicon or {}, edges or {}
        return self

    def predict(self, tokens, precomputed_X=None):
        validate_token_sequences(tokens)
        if not self._is_fitted:
            raise RuntimeError("The action-ending CRF must be fitted first.")
        if self.profile_source_sha256_ != sha256_file(Path(__file__)):
            raise ValueError("Action-ending feature source changed; retrain the isolated candidate.")
        if precomputed_X is not None:
            raise ValueError("This profile constructs its own matrices.")
        indices, output = [i for i, sentence in enumerate(tokens) if sentence], [[] for _ in tokens]
        if indices:
            words = [tokens[i] for i in indices]
            lexicon = self.lexicon_ if self.feature_config.get("use_action_lexicon", False) else None
            matrix = [ending_features(sentence,
                action_features(sentence, _sentence_features(sentence, **self.base_feature_config),
                                self.feature_config.get("use_action_morphology", False), lexicon),
                self.edge_lexicon_ if self.learned_edges else None) for sentence in words]
            for i, sequence in zip(indices, CRFTagger.predict(self, words, precomputed_X=matrix)):
                output[i] = sequence
        return output


def save_candidate(model, path, metadata):
    if not isinstance(model, ActionEndCRFTagger):
        raise ValueError("Expected an isolated action-ending candidate.")
    details = {**metadata, "ending_profile": {"version": PROFILE_VERSION, "learned_edges": model.learned_edges,
               "source_sha256_lf": sha256_file(Path(__file__)), "requires_guarded_loader": True}}
    return save_ner_tagger(model, path, "action", metadata=details)


def load_candidate(path):
    artifact = load_ner_tagger(path, expected_layer="action")
    profile = artifact.manifest["metadata"].get("ending_profile", {})
    if (not isinstance(artifact.tagger, ActionEndCRFTagger) or profile.get("version") != PROFILE_VERSION
            or profile.get("source_sha256_lf") != sha256_file(Path(__file__))
            or profile.get("learned_edges") is not artifact.tagger.learned_edges
            or profile.get("requires_guarded_loader") is not True):
        raise ValueError("Action-ending candidate profile/source changed; regenerate the isolated artifact.")
    return artifact
