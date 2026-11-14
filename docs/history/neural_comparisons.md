# Excluded historical neural experiments

These figures were recorded before the user clarified the no-deep-learning
constraint on 2026-11-15. They are preserved for provenance and are excluded
from active comparisons, tests, future training, and project model selection.
They have not been rerun or revalidated for this scope change.

spaCy's trained NER component is neural. Its blank tokenizer and dictionary
matcher used by the classical pipeline are separate allowed utilities.

## Earlier relevant-only-training entity dev results

| Historical model | All-dev P | All-dev R | All-dev F1 | Relevant P | Relevant R | Relevant F1 |
|---|---:|---:|---:|---:|---:|---:|
| spaCy NER | 0.199 | 0.332 | 0.249 | 0.386 | 0.332 | 0.357 |
| BiLSTM-CRF | 0.182 | 0.212 | 0.196 | 0.350 | 0.212 | 0.264 |

| Historical model | THREAT_ACTOR | MALWARE_TOOL | INFRASTRUCTURE |
|---|---:|---:|---:|
| spaCy NER | 0.522 | 0.335 | 0.285 |
| BiLSTM-CRF | 0.455 | 0.174 | 0.233 |

## Earlier relevant-only-training action dev results

| Historical model | All-dev P | All-dev R | All-dev F1 | Relevant P | Relevant R | Relevant F1 |
|---|---:|---:|---:|---:|---:|---:|
| spaCy NER | 0.096 | 0.202 | 0.130 | 0.205 | 0.202 | 0.204 |
| BiLSTM-CRF | 0.122 | 0.161 | 0.139 | 0.221 | 0.161 | 0.186 |

An older all-sentence-training DistilBERT experiment recorded entity dev F1
0.245 and approximately 2.5 hours of CPU training. This is a different setup
from the relevant-only rows above and is not an active upgrade path.

The original experiment source files are retained with import-time scope guards:
`src/cti/models/bert_ner.py`, `bilstm_crf.py`, and `spacy_ner.py`. Their former
dependency list is archived as `excluded_neural_requirements.txt` in this folder.
