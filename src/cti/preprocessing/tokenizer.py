import spacy

_CACHE = {}
_LANGUAGE_FALLBACKS = {"gu", "mr"}


def _get_nlp(lang: str):
    if not isinstance(lang, str) or not lang:
        raise ValueError("lang must be a non-empty spaCy language code.")
    if lang not in _CACHE:
        try:
            _CACHE[lang] = spacy.blank(lang)
        except (ImportError, ValueError):
            if lang not in _LANGUAGE_FALLBACKS:
                raise
            _CACHE[lang] = spacy.blank("xx")
    return _CACHE[lang]


def tokenize(text: str, lang: str = "en"):
    """Return a list of (token, start, end) using spaCy's tokenizer."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    doc = _get_nlp(lang)(text)
    return [(t.text, t.idx, t.idx + len(t.text)) for t in doc if not t.is_space]