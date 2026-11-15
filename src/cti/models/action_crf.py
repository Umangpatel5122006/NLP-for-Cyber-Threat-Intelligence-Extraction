"""Optional classical action-boundary features over the existing CRF.

Morphology/function-word cues are fixed rules, not a trained POS parser.
Boundary statistics are learned from fit inputs only. Training features leave
out the current sentence's counts; prediction uses the full fitted dictionary.
"""

import math
import string
from collections import defaultdict

from cti.models.base import validate_token_sequences, validate_training_data
from cti.models.crf import CRFTagger, _EDGE_PUNCTUATION, _sentence_features


PROFILE_FLAGS = ("use_action_morphology", "use_action_lexicon")
VERBS = frozenset("""
access abuse allow attempt bypass collect communicate connect copy create decode
decrypt delete deploy disable download drop elevate enable encode encrypt enumerate
escalate establish evade execute exfiltrate exploit extract impersonate inject install
kill launch leverage load maintain modify move obtain open perform persist read
receive register remove retrieve run scan search send spawn start steal stop target
upload use write
""".split())
IRREGULAR = {
    "ran": "run", "sent": "send", "stole": "steal", "stolen": "steal",
    "wrote": "write", "written": "write", "read": "read", "found": "find",
    "made": "make", "built": "build", "took": "take", "taken": "take",
    "left": "leave", "led": "lead", "was": "be", "were": "be", "been": "be",
    "being": "be", "has": "have", "had": "have", "does": "do", "did": "do",
    "done": "do", "is": "be", "are": "be",
}
FUNCTION_WORDS = {
    "aux": frozenset("am is are was were be been being do does did have has had".split()),
    "modal": frozenset("can could may might must shall should will would".split()),
    "neg": frozenset(["no", "not", "never", "without", "n't"]),
    "prep": frozenset("to from in on at by for with via into through over under of as after before during between against upon".split()),
    "conj": frozenset("and or but nor yet while whereas because although when if that which until unless once".split()),
    "det": frozenset("a an the this that these those each any some".split()),
    "pron": frozenset("it they he she we them its their".split()),
}


def _verb_forms():
    forms = {}
    for verb in sorted(VERBS):
        forms[verb] = verb
        if verb.endswith("y") and verb[-2] not in "aeiou":
            forms[verb[:-1] + "ies"] = verb
            forms[verb[:-1] + "ied"] = verb
        else:
            forms[verb + ("es" if verb.endswith(("s", "x", "z", "ch", "sh", "o")) else "s")] = verb
            irregular_past = {"run": "ran", "send": "sent", "steal": "stole", "read": "read", "write": "wrote"}
            if verb in irregular_past:
                past = irregular_past[verb]
            elif verb.endswith("e"):
                past = verb + "d"
            elif verb in ("drop", "stop", "scan"):
                past = verb + verb[-1] + "ed"
            else:
                past = verb + "ed"
            forms[past] = verb
        progressive = verb[:-1] + "ing" if verb.endswith("e") else verb + (verb[-1] if verb in ("drop", "stop", "scan", "run") else "") + "ing"
        forms[progressive] = verb
    forms.update(IRREGULAR)
    return forms


VERB_FORMS = _verb_forms()


def _clean(word):
    return word.lower().strip(_EDGE_PUNCTUATION)


def action_stem(word):
    """A small disclosed English suffix normalizer, not a full lemmatizer."""
    lower = _clean(word)
    if lower in VERB_FORMS:
        return VERB_FORMS[lower]
    if not lower.isascii() or not lower.isalpha():
        return lower
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith("ing") and len(lower) > 5:
        root = lower[:-3]
        return root[:-1] if len(root) > 2 and root[-1] == root[-2] and root[-1] in "bdgmnprt" else root
    if lower.endswith("ed") and len(lower) > 4:
        root = lower[:-2]
        return root[:-1] if len(root) > 2 and root[-1] == root[-2] and root[-1] in "bdgmnprt" else root
    if lower.endswith("s") and not lower.endswith(("ss", "us")) and len(lower) > 3:
        return lower[:-1]
    return lower


def _category(word):
    lower = _clean(word)
    for category, words in FUNCTION_WORDS.items():
        if lower in words:
            return category
    if word and all(character in string.punctuation for character in word):
        return "punct"
    return "verb_cue" if action_stem(word) in VERBS else "other"


def boundary_counts(tokens, tags):
    counts = defaultdict(lambda: [0, 0, 0, 0])
    for words, sequence in zip(tokens, tags):
        for index, (word, tag) in enumerate(zip(words, sequence)):
            stem = action_stem(word)
            if not stem:
                continue
            values = counts[stem]
            values[0] += 1
            values[1] += tag == "B-ACTION"
            values[2] += tag != "O"
            values[3] += tag != "O" and (index + 1 == len(sequence) or sequence[index + 1] != "I-ACTION")
    return {stem: tuple(values) for stem, values in counts.items()}


def action_features(words, base, morphology=True, lexicon=None, exclude=None):
    features = [dict(values) for values in base]
    stems = [action_stem(word) for word in words]
    categories = [_category(word) for word in words]
    for index, values in enumerate(features):
        if morphology:
            lower = _clean(words[index])
            values.update({
                "action:stem": stems[index], "action:class": categories[index],
                "action:ing": lower.endswith("ing"), "action:ed": lower.endswith("ed"),
                "action:verb_cue": stems[index] in VERBS,
            })
            for offset in (-2, -1, 1, 2):
                neighbor = index + offset
                if 0 <= neighbor < len(words):
                    values[f"action:{offset}:stem"] = stems[neighbor]
                    values[f"action:{offset}:class"] = categories[neighbor]
            if index:
                values["action:class_bigram"] = categories[index - 1] + "|" + categories[index]
            if index + 1 < len(words):
                values["action:next_class_bigram"] = categories[index] + "|" + categories[index + 1]
            for direction in (-1, 1):
                distance = next((step for step in range(1, 7)
                                 if not 0 <= index + direction * step < len(words)
                                 or words[index + direction * step] in (".", ",", ";", ":", "!", "?")), 7)
                values[f"action:{direction}:clause_distance"] = distance
        if lexicon is not None:
            for offset in (-1, 0, 1):
                neighbor = index + offset
                if not 0 <= neighbor < len(words):
                    continue
                stem = stems[neighbor]
                counts = lexicon.get(stem, (0, 0, 0, 0))
                removed = (exclude or {}).get(stem, (0, 0, 0, 0))
                total, starts, inside, ends = [a - b for a, b in zip(counts, removed)]
                known = total >= 5
                prefix = f"action:lex:{offset}:"
                values[prefix + "known"] = known
                if known:
                    values[prefix + "log_count"] = math.log1p(total)
                    for name, count in (("start", starts), ("inside", inside), ("end", ends)):
                        values[prefix + name] = (count + .5) / (total + 1)
    return features


class ActionBoundaryCRFTagger(CRFTagger):
    """The same linear-chain CRF with optional action-specific input features."""

    def __init__(self, algorithm="lbfgs", c1=1.0, c2=1.0, max_iterations=200,
                 min_freq=0, all_possible_transitions=True, feature_config=None, **kwargs):
        config = dict(feature_config or {})
        for flag in PROFILE_FLAGS:
            if type(config.get(flag, False)) is not bool:
                raise ValueError("Action feature flags must be boolean.")
        if not any(config.get(flag, False) for flag in PROFILE_FLAGS):
            raise ValueError("Use the base CRF when no action feature profile is enabled.")
        super().__init__(algorithm, c1, c2, max_iterations, min_freq, all_possible_transitions, config, **kwargs)
        self.base_feature_config = {key: value for key, value in config.items() if key not in PROFILE_FLAGS}
        self.lexicon_ = {}

    def fit(self, tokens, tags, dev_tokens=None, dev_tags=None, precomputed_X=None):
        validate_training_data(tokens, tags)
        if any(tag not in ("O", "B-ACTION", "I-ACTION") for sequence in tags for tag in sequence):
            raise ValueError("Action feature profiles accept the action layer only.")
        if precomputed_X is not None:
            raise ValueError("Action profiles construct their own fold-local feature matrices.")
        if (dev_tokens is None) != (dev_tags is None):
            raise ValueError("dev_tokens and dev_tags must be provided together.")
        if dev_tokens is not None:
            validate_training_data(dev_tokens, dev_tags)
        pairs = [(words, sequence) for words, sequence in zip(tokens, tags) if words]
        if not pairs:
            raise ValueError("Cannot fit action features on an empty training set.")
        words, sequences = map(list, zip(*pairs))
        lexicon = boundary_counts(words, sequences) if self.feature_config.get("use_action_lexicon", False) else None
        matrix = []
        for sentence, sequence in pairs:
            excluded = boundary_counts([sentence], [sequence]) if lexicon is not None else None
            matrix.append(action_features(
                sentence, _sentence_features(sentence, **self.base_feature_config),
                self.feature_config.get("use_action_morphology", False), lexicon, excluded,
            ))
        super().fit(words, sequences, precomputed_X=matrix)
        self.lexicon_ = lexicon or {}
        return self

    def predict(self, tokens, precomputed_X=None):
        validate_token_sequences(tokens)
        if not self._is_fitted:
            raise RuntimeError("The action CRF must be fitted before prediction.")
        if precomputed_X is not None:
            raise ValueError("Action profiles construct their own feature matrices.")
        indices = [index for index, sentence in enumerate(tokens) if sentence]
        output = [[] for _ in tokens]
        if indices:
            words = [tokens[index] for index in indices]
            lexicon = self.lexicon_ if self.feature_config.get("use_action_lexicon", False) else None
            matrix = [action_features(
                sentence, _sentence_features(sentence, **self.base_feature_config),
                self.feature_config.get("use_action_morphology", False), lexicon,
            ) for sentence in words]
            for index, sequence in zip(indices, super().predict(words, precomputed_X=matrix)):
                output[index] = sequence
        return output


def create_crf_tagger(**parameters):
    config = parameters.get("feature_config") or {}
    for flag in PROFILE_FLAGS:
        if type(config.get(flag, False)) is not bool:
            raise ValueError("Action feature flags must be boolean.")
    if any(config.get(flag, False) for flag in PROFILE_FLAGS):
        return ActionBoundaryCRFTagger(**parameters)
    if any(flag in config for flag in PROFILE_FLAGS):
        parameters["feature_config"] = {key: value for key, value in config.items() if key not in PROFILE_FLAGS}
    return CRFTagger(**parameters)
