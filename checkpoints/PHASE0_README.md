# Phase 0 - Setup and Plug-in Architecture

Project: NLP for Cyber Threat Intelligence Extraction
Environment: Windows, Python 3.13.3

## What we did

1. **Virtual environment** (`venv`): private package box for this project.
2. **Core libraries**: scikit-learn 1.9.1, pandas 3.1.0, pyyaml.
3. **Folder layout**:
   - `data/` (raw, processed, annotated), `configs/`, `notebooks/`, `tests/`
   - `src/cti/` with `core/`, `preprocessing/`, `extractors/`, `output/`
4. **`pyproject.toml`** + `pip install -e .`: registers our `cti` package so imports work everywhere.
5. **`core/base.py`**: the shared contract.
   - `Entity`: one finding (text, label, start, end, source, confidence)
   - `Document`: the report plus everything found so far
   - `BaseModule`: every module must have `process(doc) -> doc`
6. **`core/registry.py`**: phone book of modules. `@register("name")` signs a module up, `create("name")` builds it.
7. **`configs/config.yaml`**: control panel that lists which modules run, in order.
8. **`core/pipeline.py`**: reads the config, builds modules by name, runs a report through them.
9. **`extractors/dummy.py`**: toy module used to prove the whole system works.
10. **Tests**: `tests/test_base.py`, `tests/test_pipeline.py`.

## Key idea

Adding or swapping a model = write one file + change one line in `config.yaml`.
No other code changes.

## Files created

```
configs/config.yaml
pyproject.toml
src/cti/core/base.py
src/cti/core/registry.py
src/cti/core/pipeline.py
src/cti/extractors/dummy.py
tests/test_base.py
tests/test_pipeline.py
```

## Next: Phase 1 - Data

Collect sample threat reports and define the entity labels.
