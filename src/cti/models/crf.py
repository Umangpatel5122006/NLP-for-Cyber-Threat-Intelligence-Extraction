import json
from pathlib import Path

from spacy.matcher import PhraseMatcher
import ipaddress
import re

import spacy
import sklearn_crfsuite
from spacy.tokens import Doc

from cti.core.registry import register
from cti.models.base import BaseTagger


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

_EDGE_PUNCTUATION = ".,;:!?()[]{}<>\"'"


# ---------------------------------------------------------------------
# Cybersecurity vocabulary
# ---------------------------------------------------------------------

TRIGGERS = {
    "malware", "ransomware", "trojan", "backdoor", "botnet",
    "loader", "implant", "tool", "exploit", "group", "actor",
    "actors", "adversary", "adversaries", "attacker", "attackers",
    "apt", "campaign", "gang", "operators", "family",
}

FILE_EXT = {
    "dll", "exe", "sys", "bat", "ps1", "vbs", "js", "jar",
    "doc", "docx", "xls", "xlsx", "zip", "rar", "lnk", "tmp",
    "dat", "bin", "so", "sh", "msi", "hta", "scr", "pif",
    "iso", "img", "cab", "7z", "pdf", "conf", "cfg", "ini",
}


_GAZETTEER_PATH = Path("data/external/gazetteer.json")
_MATCHER = None


def _get_matcher():
    """Build the dictionary matcher once, on first use."""
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = PhraseMatcher(_NLP.vocab, attr="LOWER")
        if _GAZETTEER_PATH.exists():
            with open(_GAZETTEER_PATH, encoding="utf-8") as f:
                for label, names in json.load(f).items():
                    _MATCHER.add(label, [_NLP.make_doc(n) for n in names])
    return _MATCHER


def _gazetteer_tags(doc):
    """For each token: {label: 'B' or 'I'} if it is inside a known name."""
    tags = [dict() for _ in doc]
    for match_id, start, end in _get_matcher()(doc):
        label = _NLP.vocab.strings[match_id]
        for t in range(start, end):
            tags[t][label] = "B" if t == start else "I"
    return tags

def _looks_ipv6(value):
    """Return True if the token contains a valid IPv6 address."""
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
        "prefix1": lower[:1],
        "prefix2": lower[:2],
        "prefix3": lower[:3],
        "prefix4": lower[:4],
        "suffix1": lower[-1:],
        "suffix2": lower[-2:],
        "suffix3": lower[-3:],
        "suffix4": lower[-4:],
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
        "trigger": lower in TRIGGERS,
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

    # spaCy is used only for token shapes, not linguistic parsing.
    doc = Doc(_NLP.vocab, words=words)
    shapes = [token.shape_ for token in doc]
    gaz = _gazetteer_tags(doc)

    metadata = [
        _token_metadata(word)
        for word in words
    ]

    features = []

    for i, word in enumerate(words):
        current = metadata[i]

        # Keep the original feature names for compatibility.
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

        # Dictionary clues: is this token inside a known actor/malware name?
        for label in ("THREAT_ACTOR", "MALWARE_TOOL"):
            f[f"gaz:{label}"] = gaz[i].get(label, "O")
            for d in (-1, 1):
                j = i + d
                if 0 <= j < len(words):
                    f[f"{d}:gaz:{label}"] = gaz[j].get(label, "O")

        # Context window: two tokens before and after the current token.
        for d in (-2, -1, 1, 2):
            j = i + d

            if 0 <= j < len(words):
                neighbor = words[j]
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
            else:
                f["BOS" if d < 0 else "EOS"] = True

        # Preserve the original previous-word/current-word feature.
        if i > 0:
            f["bigram"] = (
                metadata[i - 1]["lower"]
                + "|"
                + current["lower"]
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
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    def fit(self, tokens, tags, dev_tokens=None, dev_tags=None):
        """
        Train the CRF model.

        tokens: list of tokenized sentences
        tags:   list of corresponding tag sequences

        dev_tokens and dev_tags are retained for API compatibility.
        They are not currently used for training or model selection.
        """
        if len(tokens) != len(tags):
            raise ValueError(
                "tokens and tags must contain the same number "
                "of sentences."
            )

        for i, (sentence, sentence_tags) in enumerate(
            zip(tokens, tags)
        ):
            if len(sentence) != len(sentence_tags):
                raise ValueError(
                    f"Sentence {i}: token count ({len(sentence)}) "
                    f"does not match tag count ({len(sentence_tags)})."
                )

        X_train = [
            _sentence_features(sentence)
            for sentence in tokens
        ]

        self.model.fit(X_train, tags)

    def predict(self, tokens):
        """
        Predict tags for tokenized sentences.

        Returns:
            A list of tag sequences, preserving sentence boundaries.
        """
        X_test = [
            _sentence_features(sentence)
            for sentence in tokens
        ]

        return self.model.predict(X_test)




