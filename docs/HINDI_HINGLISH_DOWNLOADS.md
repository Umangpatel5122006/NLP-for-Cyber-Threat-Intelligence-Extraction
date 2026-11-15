# Hindi/Hinglish download shortlist

Checked 2026-11-15. Keep original filenames and accompanying README/license
files in **`data/local_sms/incoming/`**, which is ignored by Git. Download data
files, not pretrained model weights or another project's executable code.

Local receipt (2026-11-15): the user's downloaded `data.json` and source README
are copied byte-for-byte to `data/local_sms/incoming/bolewara_hinglish/`.
The JSON SHA-256 is
`4c2ea6d6c9303f6b7e27740cb3bbebde57ec3cc10ec7b7f9e565e99b5d483f16`.
Schema checks pass for 3,787 text/label records (3,040 ham, 747 spam), with
3,558 distinct exact texts. There are no per-row language/origin fields. The
existing protected-text audit excludes 3,700 canonical and one near match,
leaving 86 candidate rows for further review; these are not all necessarily
Hinglish or independently collected. The ignored `audit.json` records details.
No model was trained on this receipt. The user reports the Hindi Dataport
download is access-restricted, so no new Hindi corpus has been imported.

| Target | Link | What it provides / current use |
|---|---|---|
| Hindi, first choice | [Spam SMS in Hindi Language — IEEE Dataport](https://doi.org/10.21227/5y8x-n678) | Creators describe 3,894 collected Hindi messages with spam/ham labels. Download access could require an IEEE account/institution. File/schema/license still need inspection after access. |
| Hinglish diagnostic supplement | [Direct data.json](https://huggingface.co/bolewara/hinglish-scam-text-dataset/resolve/main/data.json?download=true), [source card](https://huggingface.co/bolewara/hinglish-scam-text-dataset/blob/main/README.md) | Card declares CC BY 4.0, ~3,700 English examples derived/curated from UCI, and only 35 scam + 28 benign **hand-written** Hinglish examples. Diagnostic/possible separately disclosed weak-data research; not independent real-SMS ground truth. |

The Hindi source is described by its creators on
[their dataset page](https://sites.google.com/cse.nits.ac.in/ramanujamge/publications/datasets-offered).
We could read that description, but not verify a Dataport file download in this
environment. Keep the license and original source partition information for review.
Spam/ham labels are not the same as verified phishing/fraud outcomes.

The Hinglish JSON uses a `rows` list of `text`/`label` records, according to its
card. Its UCI-derived English rows can overlap our reserved corpus. Do not merge
or randomly resplit them into training. Its authored Hinglish examples are not
eligible for the current real-data-only intake and cannot establish real-world
accuracy. A larger genuinely collected Hinglish labeled set is still needed.

Two superficially promising Kaggle results were also checked but are not
recommended as Hinglish corpora: the 3,200-row Indian social-media fraud CSV
marks every row's language English, and the 2,267-row India Spam SMS collection
contains no Devanagari rows and no Hinglish matches under the current heuristic.
Those observations are data checks, not verified language annotation. Dataset
titles alone are insufficient evidence of Hindi/Hinglish coverage.

After downloading, provide the filenames or let the next repository review
inspect this folder. We will audit provenance, labels, source reuse and grouped
splits before any classical TF-IDF/linear training. The current app integration
already runs Hindi/Hinglish phrase rules; its trained models remain English-only.
