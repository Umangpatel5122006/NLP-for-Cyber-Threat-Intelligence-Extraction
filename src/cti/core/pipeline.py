"""The runner reads config.yaml, looks each module up in the registry,
 builds them, and runs a report through them in order. 
 Modules preserve the supplied report and track text changes before extraction.
"""

import importlib
import pkgutil

import yaml

from cti.core.base import Document
from cti.core.model_policy import EXCLUDED_NER_MODELS
from cti.core.registry import create


def load_all_modules():
    """Explicitly import in-scope modules, skipping historical neural models.

    Configured document pipelines use lazy built-in registration instead.
    """
    for pkg_name in ["cti.preprocessing", "cti.extractors", "cti.models", "cti.output"]:
        pkg = importlib.import_module(pkg_name)
        for m in pkgutil.iter_modules(pkg.__path__):
            if pkg_name == "cti.models" and m.name in EXCLUDED_NER_MODELS:
                continue
            importlib.import_module(f"{pkg_name}.{m.name}")


class Pipeline:
    def __init__(self, modules):
        self.modules = list(modules)
        finalizers = [index for index, module in enumerate(self.modules) if module.name == "result_merger"]
        if finalizers and finalizers != [len(self.modules) - 1]:
            raise ValueError("result_merger must run once, after all extraction modules.")

    @classmethod
    def from_config(cls, path: str):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)    # 2. read the plan  , contain like this {'pipeline': list[{'module': 'dummy', 'params': {}}]}
        modules = [
            create(step["module"], **step.get("params", {}))
            for step in cfg["pipeline"]          # 3. build each worker by name
        ]
        return cls(modules) # 4. return the finished pipeline

    def run(self, text: str, *, language_hint="auto") -> Document:
        if language_hint not in ("auto", "en", "hi", "hi-en"):
            raise ValueError("Choose auto, en, hi or hi-en as the language setting.")
        doc = Document(text=text)  # wrap the text in a Document
        if language_hint != "auto":
            doc.metadata["language_hint"] = language_hint
        for module in self.modules:
            doc = module.process(doc)   # each worker adds its findings
            doc.validate_text_tracking()
            if doc.original_text != text:
                raise ValueError("Pipeline modules must preserve the original report text.")
        return doc.attach_original_offsets()
