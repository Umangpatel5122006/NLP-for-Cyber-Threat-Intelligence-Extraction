import importlib

from cti.core.base import BaseModule
from cti.core.model_policy import check_model_scope

_REGISTRY: dict[str, type[BaseModule]] = {}

# Resolve built-in document modules without importing unrelated model packages.
_BUILTIN_MODULES = {
    "cleaner": "cti.preprocessing.cleaner",
    "defanger": "cti.preprocessing.defanger",
    "language_detector": "cti.preprocessing.language_detector",
    "sentence_splitter": "cti.preprocessing.sentence_splitter",
    "ioc_finder": "cti.extractors.ioc_finder",
    "ioc_regex": "cti.extractors.ioc_regex",
    "message_risk": "cti.extractors.message_risk",
    "dummy": "cti.extractors.dummy",
    "relevance_filter": "cti.models.relevance",
    "ner_entities": "cti.extractors.ner",
    "ner_action": "cti.extractors.ner",
    "result_merger": "cti.output.merger",
}


def register(name: str):
    """Decorator: sign a module up in the phone book under `name`."""
    check_model_scope(name)
    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls
    return wrapper


def create(name: str, **kwargs) -> BaseModule:
    """Look a module up by name and build it."""
    check_model_scope(name)
    if name not in _REGISTRY and name in _BUILTIN_MODULES:
        importlib.import_module(_BUILTIN_MODULES[name])
    if name not in _REGISTRY:
        raise KeyError(f"No module named '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
