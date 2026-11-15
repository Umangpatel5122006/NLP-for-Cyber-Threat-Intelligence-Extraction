# Collecting the next SMS training batch

The current expanded model catches more positives but also flags legitimate
notifications. Its default activation remains held. The next useful input is
**new, independently reviewed real messages**, especially legitimate banking
credits, delivery updates, job communications and anti-scam warnings, alongside
diverse actual scams. Rewording the already reported example is diagnostic work,
not a new independent training dataset.

## Start here

An empty local template is already prepared at `data/local_sms/annotations.csv`.
This directory is ignored by Git. On a fresh clone create it with:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cti.sms_intake init
```

Fill it using the exact columns in
[`examples/sms_annotation_template.csv`](../examples/sms_annotation_template.csv).
Save as UTF-8 CSV. Quote message fields containing commas, quotes or newlines;
normal CSV spreadsheet export handles this. The tool preserves original text.

After completing a batch, audit it from the project root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cti.sms_intake check data/local_sms/annotations.csv
```

Review `data/local_sms/batch_v1/audit.json` for label/source/category coverage
and `quarantine.json` for row IDs and exclusion reasons. These files stay local;
console output contains aggregate counts, not messages. Quarantine output omits
message text; the accepted snapshot and original CSV still contain it locally.

After reviewing the audit, freeze this batch's accepted rows and partitions:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cti.sms_intake check data/local_sms/annotations.csv --freeze
```

This command **does not train or activate a model**. It writes ignored
`accepted.jsonl` and `manifest.json` with hashes and fixed group assignments.
Repeating an unchanged frozen batch verifies it without rewriting. Changed
input or references require inspection and a new batch directory; do not rebuild
old reserved partitions. For a later batch, use a new CSV and directory:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cti.sms_intake check data/local_sms/annotations_v2.csv --output-dir data/local_sms/batch_v2 --freeze
```

The intake's hashing/grouping works in the project's native Windows environment;
it does not import spaCy or load model artifacts. The prepared WSL Python also
supports the same `-m cti.sms_intake` commands. The check requires the previously
pinned UCI and supplement downloads under `models/`. If absent, download them
in a prepared runtime:

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "from cti.core.sms_data import download_sms; from cti.core.sms_domain import download_domain; download_sms(); download_domain()"
```

## Annotation fields

| Fields | What to enter |
|---|---|
| `id`, `text` | Stable unique ID and message text; no copied diagnostic examples |
| `source_id` | Collection/dataset ID; use multiple origins when possible |
| `source_reference` | Dataset URL/row locator, or a nonidentifying private collection reference |
| `usage_permission` | `public_dataset` or `consented_research`; record the actual license/permission in the source reference or rationale |
| `observed_date` | Known observation or collection date as `YYYY-MM-DD`; do not invent an original send date |
| `language`, `origin` | `en` and `real` for eligible training rows; generated content is `synthetic` and is excluded |
| `category` | One of the categories below; category does not determine the label |
| `campaign_id` | Shared stable ID for related campaign/template variants, even when wording differs; leave blank when unknown |
| `reviewer_a`, `label_a` | First reviewer's alias and independent `ham`, `spam` or `uncertain` label |
| `reviewer_b`, `label_b` | A different reviewer's alias and independent label |
| `adjudicator`, `adjudicated_label` | Leave blank on agreement; disagreements need a third independent reviewer and an explicit final label |
| `rationale` | Brief evidence/context supporting the label and source permission; omit personal identifiers |

Categories: `financial_notification`, `delivery`, `employment`, `investment`,
`grant_or_prize`, `credential_request`, `warning`, `personal`, `other`.

Use `ham` for legitimate/nonspam communication with adequate source context;
`spam` for supported unwanted/deceptive messages. Spam includes unwanted
marketing that is not necessarily fraud. Use `uncertain` when legitimacy cannot
be established. A government title, money amount, phone number or URL alone
does not establish the label. A quoted warning about a scam has a different
purpose from a scam solicitation. Record that distinction in the rationale.

Have reviewers label without seeing model predictions. Reviewer aliases should
be distinct people, not two copies of one model or publisher label. The tool
checks recorded agreement/identities; it **cannot authenticate consent, source
truth, reviewer independence, redaction or label correctness**. Those need human
review. Do not put identifiable personal conversations in public commits.

A practical first collection target is around **200 unique reviewed templates**,
roughly balanced between legitimate and suspicious messages, with multiple
sources and varied categories. This is a collection target, not a statistical
guarantee. Avoid gathering all ham from one source and all spam from another;
`single_label_sources` highlights that confounding risk. Later training needs
source-aware validation as well as template grouping.

## What the audit and freeze enforce

- Exclude non-real, non-English, empty/oversized, uncertain or insufficiently
  reviewed records. The current classifier's scope is English messages <=1,000
  characters. Missing provenance/date/category/rationale is quarantined.
- Quarantine contradictory canonical-template labels; collapse redundant
  number/URL variants so they do not count as independent examples.
- Exclude exact/near matches to the entire pinned UCI/supplement corpora,
  authored challenge, known reported/regression messages and **all earlier
  private frozen batches**, using text only. No protected predictions or labels
  influence selection. No external message URLs are opened.
- Group connected near templates (fixed hashing cosine >=0.85) and declared
  campaign members. If any member overlaps a protected reference, exclude the
  entire connected group. Removing a duplicate alias does not break campaign
  links between its remaining members.
- Freeze five stratified group folds with seed 91: fold 0 is reserved test,
  fold 1 dev, and the other three train. Require at least five independent
  groups per label and both labels in every partition. This minimum allows
  splitting; it does not establish adequate sample size or deployment readiness.
- Verify snapshot, protocol source and prior-reference hashes. The loader
  accepts only train/dev and rejects test before file access. No official
  corpus/split or saved model is changed by the intake.

Similarity checks are conservative and cannot prove that all semantic reuse or
campaign links are known. They can exclude genuine related messages as well.
Campaign IDs and a human audit remain useful. Batch IDs, source references and
messages stay ignored; public checkpoint notes contain aggregate evidence only.
Nothing is automatically added to model training. The next model comparison
must consume only the approved new train partition, freeze its selection policy
before dev evaluation, and keep every test partition reserved.

## Public sources checked on 2026-11-15

These checks explain why another downloaded file is not automatically the missing
legitimate-message evidence. No data or model code from these sources was run.

| Source | Finding and decision for this step |
|---|---|
| [NCSU SMS Phishing release](https://github.com/wspr-ncsu/sms-phishing) | Describes phishing messages marked by VirusTotal/APWG; useful possible future positives, not the missing matched legitimate class |
| [MOZ-Smishing paper](https://aclanthology.org/2025.africanlp-1.23/) | Crowd-sourced ham/smishing, but Portuguese; outside the active English scope |
| [Indian OTP/phishing dataset card](https://huggingface.co/datasets/gandharvbakshi/SMS-dataset-OTP-OTP_INTENT_Phishing/blob/main/README.md) | Describes manual plus LLM labeling and synthetic augmentation; not accepted as independently reviewed real ground truth |
| [PayBrake V1 card](https://huggingface.co/datasets/davidbaofu/paybrake-dataset-v1) | Repackages Mishra/Soni plus IMC25 and explicitly says evaluation partitions were inspected/scored; not a fresh independent holdout for this project |

The earlier [Mishra/Soni supplement audit](SMS_DOMAIN_DATA.md) remains applicable.
Other suitable public datasets may exist; this is a reviewed shortlist, not an
exhaustive claim that none exist. A new dataset needs a source/label/overlap audit
before use, including any official partitions supplied by its authors.
