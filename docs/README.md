# Documentation guide

Start with the [project overview](../README.md), then
[installation and operating guide](SETUP.md). Use [WSL setup](WSL_SETUP.md) for
the optional Linux alternative on Windows.

| Topic | Document |
|---|---|
| Active methods, metrics and limitations | [Models and results](MODELS_AND_RESULTS.md) |
| Metric definitions and split controls | [Evaluation protocol](paper_protocol.md) |
| Active SMS corpus and partitions | [UCI provenance](SMS_DATASET.md) |
| Held expanded-data candidate | [Supplement provenance](SMS_DOMAIN_DATA.md) |
| Hindi/Hinglish sources and audit | [Indic source guide](HINDI_HINGLISH_DOWNLOADS.md) |
| Collect independently reviewed messages | [SMS collection](SMS_COLLECTION.md) |
| Resolve action-label conventions | [Annotation review](ACTION_ANNOTATION_REVIEW.md) |
| Latest implementation evidence | [Current status](../checkpoints/CURRENT_STATUS.md) |
| Tracked vs generated output | [Results policy](../results/README.md) |

`checkpoints/` and `docs/learning/M0–M3` preserve chronological experiments.
Earlier neural suggestions and runtime commands are history; current constraints
and guides supersede them. Root `learning/` contains private presentation
preparation when generated locally and is ignored by Git. It is separate from
the older versioned `docs/learning/` research notes.
