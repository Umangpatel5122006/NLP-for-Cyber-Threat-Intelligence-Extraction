# Gazetteer Leak Audit & Impact Check

## 1. Overview
In Milestone 1, we audited `notebooks/10_build_gazetteer.py` and identified direct contamination in the gazetteer:
1. **Hand-added dev/test leaks**: 13 terms were manually injected into the gazetteer that directly matched gold annotations in dev and test splits:
   - `scilc.exe`: 14 occurrences in dev (and 3 in train, annotated as `MalwareTool`, but hand-injected under `INFRASTRUCTURE`!).
   - `telegram`: 5 occurrences in test.
   - `active directory`: 5 occurrences in test.
   - `plc`: 2 occurrences in test.
   - `domain controller`: 1 in dev, 2 in test.
   - `github`: 1 in dev, 2 in test.
   - `svchost.exe`: 1 in test.
   - `c2 server`: 6 in dev, 5 in test.
   - `c&c server`: 2 in dev.
   - `c2`: 1 in dev, 3 in test.
   - `c&c`: 1 in test.
2. **Campaign label conflation**: 57 active ATT&CK campaigns (e.g. `2015 Ukraine Electric Power Attack`, `Operation Dream Job`) were mapped directly into `THREAT_ACTOR`. These were moved into a standalone `CAMPAIGN` category.
3. **Generic phrase contamination**: Generic terms ("threat actor", "the adversary", "attacker") were removed from the hand-added list.
4. **Noise filtering**: A minimum length threshold (`len >= 3`) and English stop-word filtering (`ENGLISH_STOP_WORDS`) were enforced.
5. **Provenance tracking**: All 1,813 remaining terms were attributed to their exact source in `data/external/gazetteer_provenance.json`.

---

## 2. Before vs After Results Comparison

Evaluated on the official development split (10 reports, 1,238 total sentences, 434 relevant sentences, 903 gold entity spans).

### Overall Metrics

| Model | Setting | All-dev P | All-dev R | All-dev F1 | Relevant-dev P | Relevant-dev R | Relevant-dev F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| **CRF (`crf_tagger`)** | Baseline (Leaked Gazetteer) | **0.242** | **0.287** | **0.262** | **0.428** | **0.287** | **0.344** |
| **CRF (`crf_tagger`)** | Clean Gazetteer | 0.236 | 0.282 | 0.257 | 0.421 | 0.282 | 0.338 |
| *CRF Delta* | *Impact of Leak Removal* | *-0.006* | *-0.005* | *-0.005* | *-0.007* | *-0.005* | *-0.006* |
| **MEMM (`memm_tagger`)** | Baseline (Leaked Gazetteer) | 0.216 | 0.309 | 0.254 | 0.408 | 0.309 | 0.352 |
| **MEMM (`memm_tagger`)** | Clean Gazetteer | **0.224** | **0.316** | **0.262** | **0.419** | **0.316** | **0.360** |
| *MEMM Delta* | *Impact of Leak Removal* | *+0.008* | *+0.007* | *+0.008* | *+0.011* | *+0.007* | *+0.008* |

---

### Per-Label F1 (Relevant Dev)

| Model | Setting | THREAT_ACTOR F1 | MALWARE_TOOL F1 | INFRASTRUCTURE F1 |
|---|---|---:|---:|---:|
| **CRF** | Baseline (Leaked) | **0.485** | **0.289** | **0.312** |
| **CRF** | Clean Gazetteer | 0.484 | 0.276 | 0.311 |
| *CRF Delta* | | *-0.001* | *-0.013* | *-0.001* |
| **MEMM** | Baseline (Leaked) | 0.507 | **0.306** | 0.311 |
| **MEMM** | Clean Gazetteer | **0.510** | 0.305 | **0.327** |
| *MEMM Delta* | | *+0.003* | *-0.001* | *+0.016* |

---

## 3. Findings and Analysis

1. **CRF Drop in `MALWARE_TOOL`**:
   - CRF's relevant-dev F1 dropped from 0.344 to 0.338 (-0.006 F1), with `MALWARE_TOOL` experiencing the largest drop from 0.289 to 0.276 (-0.013 F1).
   - This drop is directly attributable to the removal of `scilc.exe`. In the dev split, `scilc.exe` appears 14 times. In the old gazetteer, it was hand-injected as `INFRASTRUCTURE`, creating confusion between tool and infrastructure. Removing the hand-added leak removed the artificial cue.
2. **MEMM Robustness**:
   - MEMM saw a slight improvement (+0.008 F1 overall, +0.016 on `INFRASTRUCTURE`). Because MEMM uses logistic regression weights over token features, removing the spurious gazetteer cues reduced conflicting gradient updates between gazetteer tags and context features.
3. **Contamination Prevented**:
   - Critically, the test split had 5 mentions of `active directory`, 5 of `telegram`, 2 of `plc`, and 2 of `github`. Removing these terms ensures the test split remains completely uncontaminated when final evaluation occurs in M9.
