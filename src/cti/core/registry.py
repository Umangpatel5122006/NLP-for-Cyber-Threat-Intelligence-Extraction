from cti.core.base import BaseModule

_REGISTRY: dict[str, type[BaseModule]] = {}


def register(name: str):
    """Decorator: sign a module up in the phone book under `name`."""
    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls
    return wrapper


def create(name: str, **kwargs) -> BaseModule:
    """Look a module up by name and build it."""
    if name not in _REGISTRY:
        raise KeyError(f"No module named '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)