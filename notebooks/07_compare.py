"""Compare sequence taggers on the development split using the evaluation harness."""

import argparse
import importlib
import time
from pathlib import Path

from cti.core.registry import create
from cti.core.model_policy import CLASSICAL_NER_MODULES
from cti.eval.metrics import evaluate_predictions
from cti.models.data import load_bio

ROOT = Path(__file__).resolve().parents[1]
MODELS = CLASSICAL_NER_MODULES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", default="entities", choices=("entities", "action"))
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--models",
        default="crf_tagger",
        help="Comma-separated classical models: " + ", ".join(MODELS),
    )
    args = parser.parse_args()

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not selected_models:
        parser.error("--models must name at least one classical model.")
    for m in selected_models:
        if m not in MODELS:
            parser.error(f"Unknown model: {m}. Choose from: {list(MODELS.keys())}")

    print(f"Loading BIO datasets for layer='{args.layer}', lang='{args.lang}'...")
    train_x, train_y = load_bio(args.layer, "train", args.lang)
    dev_x, dev_y = load_bio(args.layer, "dev", args.lang)

    results = []
    for name in selected_models:
        importlib.import_module(MODELS[name])
        model = create(name)
        print(f"\n=== Training {name} on layer '{args.layer}' (all train sentences: {len(train_x)}) ===")
        t0 = time.time()
        model.fit(train_x, train_y, dev_x, dev_y)
        train_time = time.time() - t0

        pred = model.predict(dev_x)
        eval_bundle = evaluate_predictions(dev_y, pred)

        exact = eval_bundle["exact_default"]
        relaxed = eval_bundle["relaxed"]
        token = eval_bundle["token_level"]
        err = eval_bundle["error_taxonomy"]["counts"]

        results.append({
            "name": name,
            "exact_p": exact["precision"],
            "exact_r": exact["recall"],
            "exact_f1": exact["f1"],
            "relaxed_f1": relaxed["f1"],
            "token_macro_f1": token["macro_f1"],
            "train_time": train_time,
            "exact_matches": err["exact_match"],
            "boundary_errors": err["boundary_error"],
            "wrong_types": err["wrong_type"],
            "missed": err["missed"],
            "spurious": err["spurious"],
        })

    print(f"\n{'='*70}")
    print(f"SUMMARY (dev set, layer='{args.layer}')")
    print(f"{'='*70}")
    print(f"{'model':14}{'Exact P':>9}{'Exact R':>9}{'Exact F1':>10}{'Relax F1':>10}{'Tok F1':>9}{'Train s':>9}")
    for r in results:
        print(
            f"{r['name']:14}{r['exact_p']:>9.3f}{r['exact_r']:>9.3f}{r['exact_f1']:>10.3f}"
            f"{r['relaxed_f1']:>10.3f}{r['token_macro_f1']:>9.3f}{r['train_time']:>9.1f}"
        )


if __name__ == "__main__":
    main()
