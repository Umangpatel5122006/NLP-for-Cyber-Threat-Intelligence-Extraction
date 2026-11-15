# SMS corpus provenance and evaluation scope

Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection).
Citation: Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
The UCI page declares **CC BY 4.0**. The source supplies SMS ham/spam labels;
these are not general cyberattack or verified fraud labels.

Official download: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip.
The pinned `SMSSpamCollection` payload is 477,907 bytes, SHA-256
`7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`.
The code fetches only this archive, bounds its size and reads the named member
without extracting archive paths. Changed payloads are rejected for review.
Downloaded text and model binaries stay under ignored `models/`.

The parsed file has **5,574 rows**. NFKC/case normalization, URL/email/number
placeholders and punctuation normalization define **5,097 template groups**.
Seven stratified group folds, seed 42, assign fold 0 to a reserved SMS test,
fold 1 to dev, and the remaining folds to train. All original rows remain;
identical/normalized templates cannot cross partitions. This does not establish
that all semantic or near duplicates have been identified.

| Partition | Rows | Use |
|---|---:|---|
| Train | 3,981 | Three grouped CV folds; final selected fit |
| Dev | 796 | Selected model only, after selection is written |
| Test | 797 | Reserved; no predictions or metrics generated |

IDs and corpus hash are frozen in `data/splits/sms_spam_v1.json`. The development
loader rejects `test` before file access; authorization tests use synthetic data.
Split generation necessarily reads corpus labels for stratification. Train/dev
loading validates complete split coverage/group separation before returning only
the requested partition. No held-out examples were used to choose features or
thresholds. The official SMAH-CTI split files and test evaluation remain unchanged.

An English SMS baseline can miss modern, regional, multilingual or novel scams.
Spam includes unwanted content that is not fraud. Neither these labels nor a
high dev score support a promise of error-free threat detection.
