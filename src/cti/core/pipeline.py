"""The runner reads config.yaml, looks each module up in the registry,
 builds them, and runs a report through them in order. 
 After this, adding a new module only means writing a file and adding one line to the config. 
 The runner never changes again.
"""

import importlib
import pkgutil

import yaml

from cti.core.base import Document
from cti.core.registry import create


def load_all_modules():
    """Import every file in these packages so modules sign up in the registry."""
    for pkg_name in ["cti.preprocessing", "cti.extractors", "cti.models"]:
        pkg = importlib.import_module(pkg_name)
        for m in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg_name}.{m.name}")


class Pipeline:
    def __init__(self, modules):
        self.modules = modules

    @classmethod
    def from_config(cls, path: str):
        load_all_modules()     # 1. roll call, everyone signs up
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)    # 2. read the plan  , contain like this {'pipeline': list[{'module': 'dummy', 'params': {}}]}
        modules = [
            create(step["module"], **step.get("params", {}))
            for step in cfg["pipeline"]          # 3. build each worker by name
        ]
        return cls(modules) # 4. return the finished pipeline

    def run(self, text: str) -> Document:
        doc = Document(text=text)  # wrap the text in a Document
        for module in self.modules:
            doc = module.process(doc)   # each worker adds its findings
        return doc
