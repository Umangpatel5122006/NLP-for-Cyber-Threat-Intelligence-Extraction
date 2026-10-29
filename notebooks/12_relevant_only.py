r"""Compare swappable NER models and relevance filters on the dev split.

Run from the repository root:
    .\venv\Scripts\python.exe notebooks\12_relevant_only.py

Only the development split is evaluated; the test split is never loaded.
"""

import argparse
import importlib
import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from cti.core.dataset import load_smah
from cti.core.registry import create


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_MODULES = {
    "crf_tagger": "cti.models.crf",
    "memm_tagger": "cti.models.memm",
    "token_logreg": "cti.models.memm",
    "spacy_ner": "cti.models.spacy_ner",
    "bilstm_crf": "cti.models.bilstm_crf",
}


def _load_split(layer, split, raw_splits):
    """Load aligned BIO data, sentence text, and relevance labels."""
    bio_path = PROCESSED / f"bio_en_{layer}_{split}.json"
    with bio_path.open(encoding="utf-8") as file:
        records = json.load(file)

    raw_rows = raw_splits[split]
    raw_by_uid = {row["uid"]: row for _, row in raw_rows.iterrows()}
    tokens, tags, texts, relevance = [], [], [], []
    seen_uids = set()
    for record_index, record in enumerate(records):
        uid = record.get("uid")
        if uid not in raw_by_uid:
            raise ValueError(
                f"{bio_path}: record {record_index} has unknown uid {uid!r}."
            )
        if uid in seen_uids:
            raise ValueError(f"{bio_path}: duplicate uid {uid!r}.")
        seen_uids.add(uid)
        if not isinstance(record.get("tokens"), list):
            raise ValueError(f"{bio_path}: record {uid!r} has no token list.")

        sentence_tokens, sentence_tags = [], []
        for token_index, item in enumerate(record["tokens"]):
            if not isinstance(item, list) or len(item) < 4:
                raise ValueError(
                    f"{bio_path}: invalid token at record {uid!r}, index {token_index}."
                )
            sentence_tokens.append(item[0])
            sentence_tags.append(item[3])
        row = raw_by_uid[uid]
        tokens.append(sentence_tokens)
        tags.append(sentence_tags)
        texts.append(row["sentence_text"])
        relevance.append(str(row["relevant"]).casefold() == "yes")

    if seen_uids != set(raw_by_uid):
        missing = set(raw_by_uid) - seen_uids
        raise ValueError(
            f"{bio_path}: processed data is missing {len(missing)} raw sentence IDs."
        )
    return tokens, tags, texts, relevance


def _subset(tokens, tags, selector):
    return (
        [sentence for sentence, keep in zip(tokens, selector) if keep],
        [sequence for sequence, keep in zip(tags, selector) if keep],
    )


def _score(name, gold, predictions, show_report=False):
    if len(gold) != len(predictions):
        raise RuntimeError(f"{name}: prediction count does not match gold data.")
    if any(len(pred) != len(target) for pred, target in zip(predictions, gold)):
        raise RuntimeError(f"{name}: predicted sequence length does not match gold.")
    precision = precision_score(gold, predictions, zero_division=0)
    recall = recall_score(gold, predictions, zero_division=0)
    f1 = f1_score(gold, predictions, zero_division=0)
    print(f"{name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    if show_report:
        print(classification_report(gold, predictions, digits=3, zero_division=0))
    return precision, recall, f1


def _make_model(name, max_iterations, epochs):
    importlib.import_module(MODEL_MODULES[name])
    if name == "crf_tagger":
        return create(name, max_iterations=max_iterations)
    if name in ("spacy_ner", "bilstm_crf"):
        return create(name, epochs=epochs)
    return create(name)


def _relevance_models(train_texts, train_relevant):
    labels = [int(value) for value in train_relevant]
    return {
        "TF-IDF + LogisticRegression": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=1,
            ),
        ),
        "TF-IDF + LinearSVC": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LinearSVC(class_weight="balanced", random_state=1),
        ),
    }, labels


def _print_relevance_score(name, gold, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        predictions,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(gold, predictions, labels=[0, 1]).ravel()
    print(
        f"{name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
        f"(TN={tn} FP={fp} FN={fn} TP={tp})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", default="entities", choices=("entities", "action"))
    parser.add_argument(
        "--models",
        default="crf_tagger,memm_tagger,token_logreg,spacy_ner,bilstm_crf",
        help="Comma-separated registered NER model names.",
    )
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--skip-relevance", action="store_true")
    args = parser.parse_args()
    if args.max_iterations < 1 or args.epochs < 1:
        parser.error("--max-iterations and --epochs must be positive.")
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not model_names:
        parser.error("--models must name at least one model.")
    unknown = set(model_names) - set(MODEL_MODULES)
    if unknown:
        parser.error(
            f"Unknown model(s): {', '.join(sorted(unknown))}. "
            f"Choose from: {', '.join(MODEL_MODULES)}."
        )

    raw_splits = load_smah(
        raw_dir=str(ROOT / "data" / "raw"),
        splits_dir=str(ROOT / "data" / "splits"),
    )
    train_x, train_y, train_text, train_relevant = _load_split(
        args.layer, "train", raw_splits
    )
    dev_x, dev_y, dev_text, dev_relevant = _load_split(args.layer, "dev", raw_splits)
    train_rel_x, train_rel_y = _subset(train_x, train_y, train_relevant)
    dev_rel_x, dev_rel_y = _subset(dev_x, dev_y, dev_relevant)

    print(
        f"Layer={args.layer}; train relevant={len(train_rel_x)}/{len(train_x)}; "
        f"dev relevant={len(dev_rel_x)}/{len(dev_x)}; test=not loaded"
    )
    print(
        "NER models train on relevant train sentences; direct scores include "
        "all dev and relevant-only dev."
    )
    model_predictions = {}
    summary = []
    for name in model_names:
        print(f"\n=== NER: {name} ===")
        model = _make_model(name, args.max_iterations, args.epochs)
        started = time.perf_counter()
        model.fit(train_rel_x, train_rel_y)
        predictions = model.predict(dev_x)
        elapsed = time.perf_counter() - started
        all_score = _score(f"{name} / all dev ({len(dev_y)})", dev_y, predictions)
        relevant_predictions = [
            prediction for prediction, keep in zip(predictions, dev_relevant) if keep
        ]
        relevant_score = _score(
            f"{name} / relevant dev ({len(dev_rel_y)})",
            dev_rel_y,
            relevant_predictions,
            show_report=True,
        )
        model_predictions[name] = predictions
        summary.append((name, all_score[2], relevant_score[2], elapsed))

    print("\n=== NER summary (dev only) ===")
    print(f"{'model':18} {'all-dev F1':>12} {'relevant F1':>13} {'fit+predict s':>15}")
    for name, all_f1, relevant_f1, elapsed in summary:
        print(f"{name:18} {all_f1:12.3f} {relevant_f1:13.3f} {elapsed:15.1f}")

    if args.skip_relevance:
        return

    print("\n=== Relevance classifiers ===")
    relevance_models, train_binary = _relevance_models(train_text, train_relevant)
    dev_binary = [int(value) for value in dev_relevant]
    relevance_predictions = {}
    for name, classifier in relevance_models.items():
        classifier.fit(train_text, train_binary)
        predictions = classifier.predict(dev_text).tolist()
        relevance_predictions[name] = predictions
        _print_relevance_score(name, dev_binary, predictions)

    ner_for_pipeline = model_predictions.get("crf_tagger")
    if ner_for_pipeline is None:
        print("Skipping end-to-end relevance-filter scores (crf_tagger not selected).")
        return

    print("\n=== Two-stage TF-IDF relevance + CRF (all dev, end-to-end) ===")
    for name, predicted_relevance in relevance_predictions.items():
        filtered = [
            sequence if keep else ["O"] * len(sequence)
            for sequence, keep in zip(ner_for_pipeline, predicted_relevance)
        ]
        _score(name, dev_y, filtered, show_report=True)


if __name__ == "__main__":
    main()
