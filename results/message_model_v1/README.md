# Classical SMS classifier train/dev results

See [the checkpoint](../../checkpoints/INTEGRATED_FINDINGS_AND_SMS_MODEL.md) and
[data provenance](../../docs/SMS_DATASET.md).

`selection.json` records three fixed candidates, grouped training CV and the
choice frozen before dev. `dev.json` and `dev_predictions.json` record only
the selected model's dev results. `integrity_audit.json` verifies reload
predictions, independent dev rescoring, dataset/split/model hashes and grouping.
The original run log is `../message_training_run.txt`.

Selected word/character TF-IDF + linear SVM: dev spam F1 **0.9810**, 103 TP,
0 FP, 4 FN, 689 TN. This is not final-test performance or universal scam accuracy.
The user's original money/SMS message is an explicit model miss caught by rules;
see the combined API examples. No test predictions were generated.
