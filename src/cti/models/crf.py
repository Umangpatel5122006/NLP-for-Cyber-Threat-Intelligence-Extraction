import json
from pathlib import Path

from spacy.matcher import PhraseMatcher
import ipaddress
import re

import spacy
import sklearn_crfsuite
from spacy.tokens import Doc

from cti.core.registry import register
from cti.models.base import (
    BaseTagger,
    repair_bio_sequence,
    validate_token_sequences,
    validate_training_data,
)


# No external spaCy model download is required.
_NLP = spacy.blank("en")


# ---------------------------------------------------------------------
# Cybersecurity patterns
# ---------------------------------------------------------------------

_HEX = re.compile(
    r"^(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{40}|"
    r"[a-fA-F0-9]{64}|[a-fA-F0-9]{96}|"
    r"[a-fA-F0-9]{128})$"
)

_DIGITS_RE = re.compile(r"\d")
_IPV4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|1\d\d|"
    r"[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"
)

_DOMAIN_RE = re.compile(
    r"^(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
    r"[A-Z]{2,63}$",
    re.IGNORECASE,
)

_URL_RE = re.compile(
    r"^(?:https?://|ftp://|www\.)\S+$",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(
    r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}$",
    re.IGNORECASE,
)

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

_REGISTRY_RE = re.compile(
    r"^HKEY_(?:CLASSES_ROOT|CURRENT_USER|LOCAL_MACHINE|"
    r"USERS|CURRENT_CONFIG)",
    re.IGNORECASE,
)

_WINDOWS_PATH_RE = re.compile(r"^(?:[A-Z]:\\|\\\\)", re.IGNORECASE)

_EDGE_PUNCTUATION = ".,;:!?()[]{}<>\"'`\u201c\u201d\u2014-"


# ---------------------------------------------------------------------
# Cybersecurity vocabulary
# ---------------------------------------------------------------------

TRIGGERS = {
    "malware", "ransomware", "trojan", "backdoor", "botnet",
    "loader", "implant", "tool", "exploit", "group", "actor",
    "actors", "adversary", "adversaries", "attacker", "attackers",
    "apt", "campaign", "gang", "operators", "family", "c2", "server",
}

FILE_EXT = {
    "dll", "exe", "sys", "bat", "ps1", "vbs", "js", "jar",
    "doc", "docx", "xls", "xlsx", "zip", "rar", "lnk", "tmp",
    "dat", "bin", "so", "sh", "msi", "hta", "scr", "pif",
    "iso", "img", "cab", "7z", "pdf", "conf", "cfg", "ini",
}


# Resolve from the project root so model results do not depend on the caller's
# current working directory.
_GAZETTEER_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "external" / "gazetteer.json"
)
_MATCHER = None


def _get_matcher():
    """Build the dictionary matcher once, on first use."""
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = PhraseMatcher(_NLP.vocab, attr="LOWER")
        if _GAZETTEER_PATH.exists():
            with open(_GAZETTEER_PATH, encoding="utf-8") as f:
                gazetteer = json.load(f)
            if not isinstance(gazetteer, dict):
                raise ValueError(f"Gazetteer must be a mapping: {_GAZETTEER_PATH}")
            for label, names in gazetteer.items():
                if not isinstance(label, str) or not isinstance(names, list):
                    raise ValueError(
                        f"Invalid gazetteer entry for label {label!r}."
                    )
                if any(not isinstance(name, str) or not name for name in names):
                    raise ValueError(
                        f"Gazetteer names for {label!r} must be non-empty strings."
                    )
                patterns = [_NLP.make_doc(name) for name in names if name]
                if patterns:
                    _MATCHER.add(label, patterns)
    return _MATCHER


def _gazetteer_tags(doc):
    """For each token: {label: 'B' or 'I'} if it is inside a known name."""
    tags = [dict() for _ in doc]
    for match_id, start, end in _get_matcher()(doc):
        label = _NLP.vocab.strings[match_id]
        for t in range(start, end):
            if label not in tags[t]:
                tags[t][label] = "B" if t == start else "I"
    return tags

def _looks_ipv6(value):
    """Return True if the token contains a valid IPv6 address."""
    if ":" not in value or value.count(":") < 2:
        return False
    try:
        return ipaddress.ip_address(value).version == 6
    except ValueError:
        return False


def _token_metadata(word):
    """
    Calculate reusable lexical and cybersecurity features
    for a single token.
    """
    lower = word.lower()
    clean = word.strip(_EDGE_PUNCTUATION)
    clean_lower = clean.lower()

    ext = (
        clean_lower.rsplit(".", 1)[-1]
        if "." in clean_lower
        else ""
    )

    looks_hash = bool(
        _HEX.fullmatch(word) or _HEX.fullmatch(clean)
    )

    looks_ipv4 = bool(_IPV4_RE.fullmatch(clean))
    looks_ipv6 = _looks_ipv6(clean)

    looks_url = bool(_URL_RE.fullmatch(clean))
    looks_email = bool(_EMAIL_RE.fullmatch(clean))
    looks_domain = bool(_DOMAIN_RE.fullmatch(clean))
    looks_cve = bool(_CVE_RE.fullmatch(clean))

    looks_registry = bool(_REGISTRY_RE.match(clean))
    looks_path = (
        bool(_WINDOWS_PATH_RE.match(clean))
        or "/" in word
        or "\\" in word
    )

    looks_ioc = any(
        (
            looks_hash,
            looks_ipv4,
            looks_ipv6,
            looks_url,
            looks_email,
            looks_domain,
            looks_cve,
        )
    )

    return {
        "lower": lower,
        "clean_lower": clean_lower,
        "normalized": _DIGITS_RE.sub("0", lower),
        "prefix1": clean_lower[:1] if clean_lower else lower[:1],
        "prefix2": clean_lower[:2] if clean_lower else lower[:2],
        "prefix3": clean_lower[:3] if clean_lower else lower[:3],
        "prefix4": clean_lower[:4] if clean_lower else lower[:4],
        "suffix1": clean_lower[-1:] if clean_lower else lower[-1:],
        "suffix2": clean_lower[-2:] if clean_lower else lower[-2:],
        "suffix3": clean_lower[-3:] if clean_lower else lower[-3:],
        "suffix4": clean_lower[-4:] if clean_lower else lower[-4:],
        "is_upper": word.isupper(),
        "is_title": word.istitle(),
        "is_alpha": word.isalpha(),
        "is_digit": word.isdigit(),
        "is_alnum": word.isalnum(),
        "has_digit": any(char.isdigit() for char in word),
        "has_hyphen": "-" in word,
        "has_dot": "." in word,
        "has_slash": "/" in word,
        "has_backslash": "\\" in word,
        "has_underscore": "_" in word,
        "has_colon": ":" in word,
        "has_at": "@" in word,
        "has_internal_capital": any(
            char.isupper() for char in word[1:]
        ),
        "has_non_ascii": any(
            ord(char) > 127 for char in word
        ),
        "looks_hash": looks_hash,
        "looks_ipv4": looks_ipv4,
        "looks_ipv6": looks_ipv6,
        "looks_url": looks_url,
        "looks_email": looks_email,
        "looks_domain": looks_domain,
        "looks_cve": looks_cve,
        "looks_registry": looks_registry,
        "looks_path": looks_path,
        "looks_ioc": looks_ioc,
        "file_ext": ext in FILE_EXT,
        "trigger": lower in TRIGGERS or clean_lower in TRIGGERS,
        "length": min(len(word), 15),
    }


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def _sentence_features(words):
    """
    Convert a tokenized sentence into CRF feature dictionaries.

    Input:
        ["Detected", "Emotet", "on", "host.exe"]

    Output:
        [
            {"lower": "detected", ...},
            {"lower": "emotet", ...},
            ...
        ]
    """
    words = list(words)

    if not words:
        return []

    # spaCy is used for token shapes and phrase matching.
    doc = Doc(_NLP.vocab, words=words)
    shapes = [token.shape_ for token in doc]
    gaz = _gazetteer_tags(doc)

    metadata = [
        _token_metadata(word)
        for word in words
    ]

    features = []
    n = len(words)

    for i, word in enumerate(words):
        current = metadata[i]

        # Keep original feature names for compatibility.
        f = {
            "bias": 1.0,
            "lower": current["lower"],
            "shape": shapes[i],
            "suffix3": current["suffix3"],
            "suffix2": current["suffix2"],
            "prefix3": current["prefix3"],
            "is_upper": current["is_upper"],
            "is_title": current["is_title"],
            "has_digit": current["has_digit"],
            "has_hyphen": current["has_hyphen"],
            "has_dot": current["has_dot"],
            "looks_hash": current["looks_hash"],
            "file_ext": current["file_ext"],
            "trigger": current["trigger"],
            "length": current["length"],
        }

        # Additional lexical and orthographic features.
        for key in (
            "normalized",
            "prefix1",
            "prefix2",
            "prefix4",
            "suffix1",
            "suffix4",
            "is_alpha",
            "is_digit",
            "is_alnum",
            "has_slash",
            "has_backslash",
            "has_underscore",
            "has_colon",
            "has_at",
            "has_internal_capital",
            "has_non_ascii",
        ):
            f[key] = current[key]

        # Additional cybersecurity-specific features.
        for key in (
            "looks_ipv4",
            "looks_ipv6",
            "looks_url",
            "looks_email",
            "looks_domain",
            "looks_cve",
            "looks_registry",
            "looks_path",
            "looks_ioc",
        ):
            f[key] = current[key]

        # Dictionary clues: only emit when token matches known threat actor, malware, or infrastructure.
        for label in ("THREAT_ACTOR", "MALWARE_TOOL", "INFRASTRUCTURE"):
            if label in gaz[i]:
                f[f"gaz:{label}"] = gaz[i][label]
            for d in (-1, 1):
                j = i + d
                if 0 <= j < n and label in gaz[j]:
                    f[f"{d}:gaz:{label}"] = gaz[j][label]

        # Sentence boundaries
        if i == 0:
            f["BOS"] = True
        if i == n - 1:
            f["EOS"] = True

        # Context window: two tokens before and after current token.
        for d in (-2, -1, 1, 2):
            j = i + d

            if 0 <= j < n:
                neighbor_meta = metadata[j]

                # Original contextual feature names.
                f[f"{d}:lower"] = neighbor_meta["lower"]
                f[f"{d}:shape"] = shapes[j]
                f[f"{d}:is_title"] = neighbor_meta["is_title"]
                f[f"{d}:trigger"] = neighbor_meta["trigger"]

                # Stronger contextual signals.
                f[f"{d}:prefix3"] = neighbor_meta["prefix3"]
                f[f"{d}:suffix3"] = neighbor_meta["suffix3"]
                f[f"{d}:is_upper"] = neighbor_meta["is_upper"]
                f[f"{d}:has_digit"] = neighbor_meta["has_digit"]
                f[f"{d}:file_ext"] = neighbor_meta["file_ext"]
                f[f"{d}:looks_hash"] = neighbor_meta["looks_hash"]
                f[f"{d}:looks_ioc"] = neighbor_meta["looks_ioc"]

        # Word bigram features.
        if i > 0:
            f["bigram"] = (
                metadata[i - 1]["lower"]
                + "|"
                + current["lower"]
            )
        if i < n - 1:
            f["+1:bigram"] = (
                current["lower"]
                + "|"
                + metadata[i + 1]["lower"]
            )

        features.append(f)

    return features


# ---------------------------------------------------------------------
# CRF model
# ---------------------------------------------------------------------

@register("crf_tagger")
class CRFTagger(BaseTagger):
    name = "crf_tagger"

    def __init__(self, c1=1.0, c2=1.0, max_iterations=200):
        if c1 < 0 or c2 < 0:
            raise ValueError("c1 and c2 must be non-negative.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )
        self._is_fitted = False

    def fit(self, tokens, tags, dev_tokens=None, dev_tags=None):
        """
        Train the CRF model.

        tokens: list of tokenized sentences
        tags:   list of corresponding tag sequences

        dev_tokens and dev_tags are retained for API compatibility.
        They are not currently used for training or model selection.
        """
        validate_training_data(tokens, tags)
        if (dev_tokens is None) != (dev_tags is None):
            raise ValueError("dev_tokens and dev_tags must be provided together.")
        if dev_tokens is not None:
            validate_training_data(dev_tokens, dev_tags)

        nonempty = [
            (sentence, sentence_tags)
            for sentence, sentence_tags in zip(tokens, tags)
            if sentence
        ]
        if not nonempty:
            raise ValueError("Cannot fit the CRF on an empty training set.")

        X_train = [_sentence_features(sentence) for sentence, _ in nonempty]
        y_train = [sentence_tags for _, sentence_tags in nonempty]
        self.model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, tokens):
        """
        Predict tags for tokenized sentences.

        Returns:
            A list of tag sequences, preserving sentence boundaries.
        """
        validate_token_sequences(tokens)
        if not self._is_fitted:
            raise RuntimeError("CRFTagger must be fitted before calling predict().")

        nonempty_indices = [i for i, sentence in enumerate(tokens) if sentence]
        predictions = [[] for _ in tokens]
        if nonempty_indices:
            X_test = [_sentence_features(tokens[i]) for i in nonempty_indices]
            decoded = self.model.predict(X_test)
            for index, sequence in zip(nonempty_indices, decoded):
                predictions[index] = repair_bio_sequence(sequence)
        return predictions
