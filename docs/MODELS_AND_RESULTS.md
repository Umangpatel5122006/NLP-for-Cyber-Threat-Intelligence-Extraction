# Current models, data and results

This is the deployment summary. Historical experiments do not all describe the
active models. Extraction, SMS classification and rules have different outputs
and evaluation units; there is no single combined cyber-threat accuracy.

## Active methods

| Component | Method | Why used |
|---|---|---|
| Text preparation | Unicode/whitespace cleanup, defanging and source mapping | Recognize obfuscated indicators while preserving evidence |
| Language routing | Script/Roman-Hindi heuristics; manual override | Disclose unsupported model languages |
| Entity NER | Linear-chain CRF | Joint BIO prediction from sparse lexical/context/dictionary features |
| Action NER | Separate CRF with train-derived morphology/boundary cues | Different boundaries and overlaps with actors/tools |
| SMS | Word 1–2 gram + character-within-word 3–5 gram TF-IDF; balanced LinearSVC | CPU-friendly learned lexical/spelling signal |
| Message rules | English and Hindi/Hinglish contextual phrase combinations | Inspectable credential, reward, money/request and fee evidence |
| IOC extraction | `ioc-finder` and source alignment; optional regex implementation | Structured IP/domain/URL/email/hash/CVE recognition |
| Review/merge | Conservative occurrences, evidence and typed overlaps | Avoid duplicate observations without losing disagreement |

spaCy provides blank tokenization, shapes and PhraseMatcher; no pretrained NER,
neural parser, embeddings, transformer or LLM API runs. spaCy dependencies do not
mean a neural pipeline is active. MEMM and token Logistic Regression remain
classical research alternatives; the workbench defaults to two CRFs.

`configs/ner_training.yaml` selects entity CRF lbfgs, c1=1.8427, c2=0.137,
min_freq=3, 100 iterations and possible transitions. Action CRF uses c1=c2=1,
min_freq=0, 200 iterations, possible transitions and morphology/lexicon features.
Both fit 3,006 relevant train sentences and evaluate every dev sentence.
`configs/config_ner.yaml` controls runtime order. The optional TF-IDF/Logistic
Regression relevance classifier is not a default hard gate.

## Data inventory

| Resource | Scope/status |
|---|---|
| SMAH-CTI | 100 English reports / 12,304 sentences; report splits 70 train, 10 dev, 20 test |
| CTI train | 8,635 sentences, including 3,006 relevant |
| CTI dev | 1,238 sentences, including 434 relevant |
| UCI SMS | Pinned file parses to 5,574 rows / 5,097 template groups |
| UCI partitions | Train 3,981, dev 796, test 797; grouped assignment |
| MITRE gazetteer | Tracked snapshot/alias features, not a reputation service |
| Mendeley supplement | 5,971 rows reduce to 427 after UCI overlap exclusions; candidate only |
| Supplement partitions | Train 255, dev 86, test 86 |
| Indic sources | Audited diagnostics, not supervised multilingual training |

CTI includes relevance, entity spans and ATT&CK annotations; not every annotation
is used. The workbench does not train an ATT&CK technique classifier. BIO
conversion trims/aligns raw spans to token boundaries. The UCI count is the
pinned parsed-file count; do not replace it with a different headline count.
See [UCI](SMS_DATASET.md), [supplement](SMS_DOMAIN_DATA.md) and
[Indic provenance](../checkpoints/HINDI_HINGLISH.md).

CTI partitions isolate reports. SMS partitions isolate normalized templates;
supplements/private intake have additional overlap controls. Train/dev support
development. Protected CTI/SMS test partitions remain unevaluated in the current
recorded development process. This document does not run final evaluation.

## Active development results

| Task | Precision | Recall | F1 | Evidence |
|---|---:|---:|---:|---|
| Entity CRF, all-dev exact | 0.2470 | 0.2735 | 0.2596 | [Entity report](../results/ner_v2_entities.json) |
| Action CRF, all-dev exact | 0.1419 | 0.1711 | 0.1551 | [Selected action report](../results/action_features_v1/action_boundaries_dev.json) |
| English SMS, dev spam class | 1.0000 | 0.9626 | 0.9810 | [SMS report](../results/message_model_v1/dev.json) |

SMS dev confusion matrix:

| Actual / predicted | Ham | Spam |
|---|---:|---:|
| Ham | 689 | 0 |
| Spam | 4 | 103 |

Accuracy derived from these counts is 792/796 = **99.50%**. Spam F1/recall are
more informative under class imbalance. This score is neither universal scam
accuracy nor a final-test result. The SVM margin is not fraud probability.

NER requires exact label and complete token boundaries. Entity relaxed F1 is
0.3258; action relaxed F1 is 0.4966. Relaxed overlap scores do not replace exact
extraction scores. These results use processed token-aligned BIO gold across
all 1,238 dev sentences, not a complete original-report character-span benchmark.
See [the protocol](paper_protocol.md). Optional sentence relevance scores
P=0.6265, R=0.7419, F1=0.6793; relevance does not mean maliciousness.

## SMS selection and held candidate

Train-only grouped CV compares word TF-IDF + Logistic Regression, word/character
TF-IDF + Logistic Regression and word/character TF-IDF + linear SVM. The selected
SVM is refit on train and scored on dev after selection is written. Features
learn lexical weights without rule verdicts being fed into the classifier.

A supplement-trained candidate improves supplemental dev F1 0.6400 → 0.9538,
with UCI dev F1 0.9810 → 0.9761. It is **held** because diagnostic false alarms
increase. Both models miss the supplied grant/SMS phrasing without rules; the
default remains UCI-only. See [the release hold](../checkpoints/LEARNED_MESSAGE_EXPANSION.md).

## Generalization and limitations

Rules combine concepts within bounded context rather than storing the user's
sentence. Phone targets are optional in supported offer/request patterns.
Character n-grams support lexical variations. Neither method guarantees unseen
semantic/campaign detection. Negation, warnings and legitimate notifications
need guards and can still fail.

Hindi/Hinglish have phrase rules and functional regressions only. Hand-authored
downloaded examples are development diagnostics, not independent real-world
accuracy data. Other surrounding languages can retain IOCs without gaining
semantic model support.

Selected action features improve recorded dev exact F1 0.1379 → 0.1551.
Later relevance gating, background sampling, ending features, BILOU and
consistency/deduplication experiments fail predeclared selection/confirmation
criteria and remain inactive. [Current status](../checkpoints/CURRENT_STATUS.md)
links their records.

Outstanding limits include 21 pending action-label review groups, scarce domain
data, false positives in background prose, boundary errors, language-routing
mistakes and SMS domain shift. Passing software tests does not establish flawless
detection. There is no calibrated combined probability, sender/reputation lookup,
fact checking, live network inspection or autonomous blocking.
