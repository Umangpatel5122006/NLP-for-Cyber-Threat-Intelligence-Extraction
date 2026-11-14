# Project constraints

- Use classical NLP and classical machine learning only. Deep learning is
  excluded: no BERT/transformers, BiLSTMs, neural embeddings, trained spaCy NER,
  neural dependency parsers, or LLM APIs.
- spaCy's blank tokenizer, token shape utilities, and rule-based PhraseMatcher
  remain allowed. Use CRF, MEMM, token Logistic Regression, TF-IDF, linear
  classifiers, dictionaries, and rules for the active project.
- Preserve earlier neural experiments as history; do not execute or re-enable
  them. `src/cti/core/model_policy.py` controls the supported NER choices.
  Older checkpoint/learning-note upgrade suggestions are superseded by this scope.
- Develop on train/dev only. Reserve the official test split for final frozen
  evaluation. Use synthetic data for split-lock authorization tests.
- Validate classical code without changing Windows security settings. Record
  blocked checks honestly; a prepared CI workflow is not a passing CI run.
- Work in reviewable steps. Leave commits and pushes to the user unless they
  explicitly request those actions.
