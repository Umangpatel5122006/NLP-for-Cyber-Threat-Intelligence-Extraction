# Phase 1 - Data

Project: NLP for Cyber Threat Intelligence Extraction

## What we did

1. **Defined labels** in `configs/labels.yaml`:
   - IOC labels (regex, P3): IP, DOMAIN, URL, HASH, CVE, EMAIL
   - Entity labels (NER, P5): THREAT_ACTOR, MALWARE_TOOL, ACTION, INFRASTRUCTURE
   - Later labels (no training data yet): SYSTEM, VULNERABILITY
   - `dataset_mapping`: SMAH-CTI label names -> our names
2. **Chose the dataset: SMAH-CTI**
   - 12,304 sentences, 100 reports, 4,302 relevant sentences
   - Gives NER spans, MITRE ATT&CK tactic/technique labels, relevance labels
   - Stored in `data/raw/SMAH-CTI.json`
   - License: CC BY-NC 4.0 (fine for a college project, not for commercial use)
3. **Explored the data** (`notebooks/01_explore.py`, `02_counts.py`): all counts matched the official description.
4. **Downloaded the official split** into `data/splits/`:
   - Train: 70 reports, 8,635 sentences
   - Dev: 10 reports, 1,238 sentences
   - Test: 20 reports, 2,431 sentences
   - Split is by report, so there is no leakage between parts.
5. **Wrote the data loader**: `src/cti/core/dataset.py` -> `load_smah()` returns train/dev/test tables.
6. **Reviewed label examples** (`notebooks/03_examples.py`).

## Key findings

- `Infrastructure_Indicator` mixes files, defanged domains, software, systems and generic phrases, so we kept it as one label (`INFRASTRUCTURE`) and regex will catch strict indicators separately.
- `Action` spans are long free-form phrases, so expect a lower NER score there.
- ThreatActor and MalwareTool include generic mentions ("the attackers", "the malware").
- Some spans carry stray commas or spaces (minor noise).
- Reports use defanging like `[.]`, so P2 must handle it.

## Files created

```
configs/labels.yaml
data/raw/SMAH-CTI.json
data/splits/*.json, split_statistics.csv
src/cti/core/dataset.py
notebooks/01_explore.py, 02_counts.py, 03_examples.py
tests/test_dataset.py
```

## Next: Phase 2 - Preprocessing

Cleaner, defanger, sentence splitter, language detection hook.
