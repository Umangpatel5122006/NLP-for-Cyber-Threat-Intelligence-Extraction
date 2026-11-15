# Supplemental SMS data: provenance and leakage controls

Source: Mishra, S. and Soni, D. (2022), *SMS PHISHING DATASET FOR MACHINE
LEARNING AND PATTERN RECOGNITION*, Mendeley Data, version 1,
[doi:10.17632/f45bkkt8pr.1](https://data.mendeley.com/datasets/f45bkkt8pr/1).
The publisher declares CC BY 4.0. Retain this attribution when redistributing
the data or derived work. The authors describe collecting text and screenshots
from public websites, using OCR for screenshots, and reusing UCI SMS messages.
These source labels are not independently verified fraud determinations.

The CSV contains 5,971 rows: 4,844 ham, 489 spam and 638 smishing. The publisher's
download endpoint returned HTTP 403 during this work. The reviewed copy came
from a public mirror, pinned to a Git commit:

- Repository: <https://github.com/nmbenton/INFO-4360-Project>
- Commit: `938fe00b100f88b7b9e4d0b9fb4225161f1e08ed`
- File: `data/Dataset_5971.csv` (595,987 bytes; UTF-8 BOM)
- SHA-256: `649844f1c62a6b27e145eaf17a65f7010c56c390e11a794ea8329993a05ba71e`
- Git blob SHA-1: `30aa7accdb18d8d932f296c70f95090d46eb7c39`

The Git blob hash was independently recomputed from the downloaded bytes.
Row/class counts match the publication. Publisher-hosted bytes could not be
compared because of the 403; the mirror's hash is a reproducibility pin, not a
claim that its contents were authenticated by the publisher. No code from the
mirror was executed. Downloads are bounded and changed hashes are rejected.

Only `TEXT` and the original `LABEL` are used. Binary labels map ham to ham,
and spam/smishing to spam. Supplied URL/EMAIL/PHONE flags are excluded from
features. HTML entities are decoded; OCR artifacts remain a quality limitation.
The selected raw-text model receives no rule verdict or hand-authored scam label.

## Excluding reuse of protected UCI messages

Comparing all 5,971 rows as if they were independent new data would substantially
contaminate the existing benchmark. `cti.core.sms_domain` excludes matches to
the **entire UCI corpus**, including protected partitions, using text fingerprints
only. It does not use protected labels, predictions, scores or examples for
model/threshold selection. The fixed exclusion procedure is:

1. HTML/NFKC/case normalization, URL/email/numeric placeholders, punctuation
   removal and whitespace folding; exclude canonical exact matches.
2. Stateless character 3–5 gram hashing (2^18 bins, nonnegative, L2-normalized);
   exclude a row if any protected UCI text has cosine similarity >= 0.85.

This excludes 5,124 canonical exact matches and another 420 near matches,
leaving **427 rows**: 266 ham, 53 spam, 108 smishing (401 canonical templates).
The threshold was fixed before fitting candidates. This is a conservative
overlap check, not proof that all paraphrases or shared source origins are absent.

Connected components of the same near-match relation define supplement groups.
Five stratified group folds with seed 73 assign fold 0 to test, fold 1 to dev,
and the other folds to train. The frozen manifest records row IDs, groups,
source hashes and the exclusion audit in `data/splits/sms_domain_v1.json`.

| Partition | Rows | Groups | Current use |
|---|---:|---:|---|
| Train | 255 | 225 | Grouped CV; final candidate fit |
| Dev | 86 | 75 | One selected candidate and frozen baseline comparison |
| Test | 86 | Reserved | No inference or evaluation |

Split construction reads labels for stratification. The development loader
rejects `test` before file access and verifies source hashes and group separation.
The original 797-row SMS holdout and official CTI test remain unevaluated.
Raw data stay in ignored `models/domain_research/`; split definitions are tracked.

## Limits and next data needs

After overlap removal, this is a small supplement from a related source family,
not a large independent modern fraud benchmark. Its dev positives mix ordinary
spam and smishing; report both the binary score and smishing recall. Do not call
spam F1 general cyber-threat accuracy. Manually authored diagnostic examples
are separate from training and are explicitly synthetic.

Before another model change, obtain legally shareable, labeled examples of
**legitimate** banking credits, delivery updates, job messages and anti-scam
warnings, alongside real grant/benefit, credential, delivery and investment
scams. Preserve collection source, date, labeling rationale and template groups;
remove private personal information. Do not label examples just from keywords.
Freeze new source/template partitions before tuning, and keep a separate final
evaluation set. Do not fit the current dev or diagnostic challenge examples.
The [collection and intake guide](SMS_COLLECTION.md) now supplies the private
CSV schema, review checks, protected-overlap exclusions and batch freeze commands.
