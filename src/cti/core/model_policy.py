"""The active project uses classical NLP/ML and excludes deep learning."""

CLASSICAL_NER_MODULES = {
    "crf_tagger": "cti.models.crf",
    "memm_tagger": "cti.models.memm",
    "token_logreg": "cti.models.memm",
}

EXCLUDED_NER_MODELS = frozenset({"bert_ner", "bilstm_crf", "spacy_ner"})


def check_model_scope(name):
    """Reject the known historical neural models before importing/training them."""
    if name in EXCLUDED_NER_MODELS:
        raise ValueError(
            f"{name!r} uses deep learning and is excluded from this project. "
            f"Use classical NER models: {', '.join(CLASSICAL_NER_MODULES)}."
        )
