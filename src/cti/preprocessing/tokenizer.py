import spacy

_CACHE = {}


def _get_nlp(lang: str):
    if lang not in _CACHE:
        try:
            _CACHE[lang] = spacy.blank(lang)      # tokenizer only, no model download
        except Exception:
            _CACHE[lang] = spacy.blank("xx")      # multilingual fallback
    return _CACHE[lang]


def tokenize(text: str, lang: str = "en"):
    """Return a list of (token, start, end) using spaCy's tokenizer."""
    doc = _get_nlp(lang)(text)
    return [(t.text, t.idx, t.idx + len(t.text)) for t in doc if not t.is_space]