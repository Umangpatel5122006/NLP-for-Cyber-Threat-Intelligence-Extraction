"""Stage 2: Report-grouped cross-validation of top CRF configurations.

Takes the top 5 candidate configurations from Stage 1 coarse search and evaluates
each using 5-fold cross-validation grouped strictly by document (doc_id) across all
80 train + dev reports.
Saves results to results/crf_tuning/stage2_cv_top5_{layer}.csv.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pycrfsuite

from cti.core.dataset import load_smah
from cti.eval.cv import get_report_grouped_splits
from cti.eval.metrics import exact_span_metrics, relaxed_span_metrics, token_level_metrics
from cti.models.crf import _sentence_features, repair_bio_sequence
from cti.models.data import load_bio
from cti.models.feature_cache import get_cached_sentence_features

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "crf_tuning"
CACHE_DIR = RESULTS_DIR / "cache"


def run_fold(
    config: Dict[str, Any],
    train_x_features: List[List[Dict[str, Any]]],
    train_y: List[List[str]],
    val_x_features: List[List[Dict[str, Any]]],
    val_y: List[List[str]],
    val_rel_mask: List[bool],
) -> Dict[str, float]:
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(train_x_features, train_y):
        trainer.append(xseq, yseq)

    algo = config["algorithm"]
    trainer.select(algo)
    params = {
        "max_iterations": config["max_iterations"],
        "feature.minfreq": config["min_freq"],
        "feature.possible_transitions": 1 if config["all_possible_transitions"] else 0,
    }
    if algo == "lbfgs":
        params["c1"] = config["c1"]
        params["c2"] = config["c2"]
    elif algo == "l2sgd":
        params["c2"] = config["c2"]
    trainer.set_params(params)

    tmp_path = CACHE_DIR / f"cv_temp_{time.time_ns()}.crf"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    trainer.train(str(tmp_path))

    tagger = pycrfsuite.Tagger()
    tagger.open(str(tmp_path))
    raw_preds = [tagger.tag(xseq) for xseq in val_x_features]
    preds = [repair_bio_sequence(seq) for seq in raw_preds]
    tagger.close()
    if tmp_path.exists():
        tmp_path.unlink()

    # Headline (all sentences)
    exact_all = exact_span_metrics(val_y, preds)
    relaxed_all = relaxed_span_metrics(val_y, preds)
    tok_all = token_level_metrics(val_y, preds)

    # Diagnostic (relevant only)
    rel_gold = [g for g, k in zip(val_y, val_rel_mask) if k]
    rel_pred = [p for p, k in zip(preds, val_rel_mask) if k]
    exact_rel = exact_span_metrics(rel_gold, rel_pred)

    return {
        "headline_exact_f1": exact_all["f1"],
        "headline_relaxed_f1": relaxed_all["f1"],
        "headline_token_f1": tok_all["macro_f1"],
        "diagnostic_exact_f1": exact_rel["f1"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", default="entities", choices=("entities", "action"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage1-csv",
        type=Path,
        default=None,
        help="Path to stage 1 results CSV. Defaults to results/crf_tuning/stage1_coarse_search_{layer}.csv",
    )
    args = parser.parse_args()

    stage1_csv = args.stage1_csv or (RESULTS_DIR / f"stage1_coarse_search_{args.layer}.csv")
    if not stage1_csv.exists():
        parser.error(f"Stage 1 CSV not found: {stage1_csv}")

    # Read top 5 configs from Stage 1
    with open(stage1_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Sort by all_dev_exact_f1 descending
    rows.sort(key=lambda r: float(r["all_dev_exact_f1"]), reverse=True)
    top_configs = []
    seen = set()
    for r in rows:
        key = (
            r["algorithm"], float(r["c1"]), float(r["c2"]), int(r["min_freq"]),
            r["all_possible_transitions"].lower() == "true", int(r["max_iterations"])
        )
        if key not in seen:
            seen.add(key)
            top_configs.append({
                "algorithm": r["algorithm"],
                "c1": float(r["c1"]),
                "c2": float(r["c2"]),
                "min_freq": int(r["min_freq"]),
                "all_possible_transitions": r["all_possible_transitions"].lower() == "true",
                "max_iterations": int(r["max_iterations"]),
                "stage1_dev_f1": float(r["all_dev_exact_f1"]),
            })
        if len(top_configs) == 5:
            break

    print(f"=== Stage 2: 5-Fold Grouped CV on Top {len(top_configs)} Configs ({args.layer}) ===")

    # Load 80 train + dev reports
    raw_splits = load_smah()
    train_x, train_y = load_bio(args.layer, "train")
    dev_x, dev_y = load_bio(args.layer, "dev")

    raw_train = raw_splits["train"]
    raw_dev = raw_splits["dev"]

    all_x = train_x + dev_x
    all_y = train_y + dev_y
    all_doc_ids = list(raw_train["doc_id"]) + list(raw_dev["doc_id"])
    all_rel = (
        [str(r["relevant"]).casefold() == "yes" for _, r in raw_train.iterrows()]
        + [str(r["relevant"]).casefold() == "yes" for _, r in raw_dev.iterrows()]
    )

    # Precompute / cache features for all 9,873 sentences
    cache_all = CACHE_DIR / f"feats_all_train_dev_{args.layer}.joblib"
    all_features = get_cached_sentence_features(all_x, cache_path=cache_all, verbose=True)

    # Generate 5 document-grouped folds
    splits = get_report_grouped_splits(all_doc_ids, n_splits=args.n_splits, seed=args.seed)
    print(f"Generated {len(splits)} folds across {len(set(all_doc_ids))} reports.")

    out_csv = RESULTS_DIR / f"stage2_cv_top5_{args.layer}.csv"
    fieldnames = [
        "rank", "algorithm", "c1", "c2", "min_freq", "all_possible_transitions",
        "max_iterations", "stage1_dev_f1",
        "cv_exact_f1_mean", "cv_exact_f1_std",
        "cv_relaxed_f1_mean", "cv_relaxed_f1_std",
        "cv_token_f1_mean", "cv_token_f1_std",
        "cv_diag_rel_f1_mean", "cv_diag_rel_f1_std",
        "fold_exact_f1s"
    ]

    results = []

    for idx, cfg in enumerate(top_configs, 1):
        print(
            f"\nConfig {idx}/{len(top_configs)}: {cfg['algorithm']} c1={cfg['c1']} c2={cfg['c2']} "
            f"min_freq={cfg['min_freq']} trans={cfg['all_possible_transitions']} iter={cfg['max_iterations']}"
        )
        fold_exact_f1s = []
        fold_relaxed_f1s = []
        fold_token_f1s = []
        fold_diag_f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
            # Train only on relevant sentences within the train fold
            rel_train_idx = [i for i in train_idx if all_rel[i]]
            X_tr = [all_features[i] for i in rel_train_idx]
            y_tr = [all_y[i] for i in rel_train_idx]

            # Evaluate on all sentences in val fold (and track relevance)
            X_val = [all_features[i] for i in val_idx]
            y_val = [all_y[i] for i in val_idx]
            val_rel = [all_rel[i] for i in val_idx]

            t0 = time.perf_counter()
            f_res = run_fold(cfg, X_tr, y_tr, X_val, y_val, val_rel)
            dt = time.perf_counter() - t0

            fold_exact_f1s.append(f_res["headline_exact_f1"])
            fold_relaxed_f1s.append(f_res["headline_relaxed_f1"])
            fold_token_f1s.append(f_res["headline_token_f1"])
            fold_diag_f1s.append(f_res["diagnostic_exact_f1"])

            print(
                f"  Fold {fold_idx}: Exact F1={f_res['headline_exact_f1']:.3f} "
                f"Relaxed F1={f_res['headline_relaxed_f1']:.3f} "
                f"Token F1={f_res['headline_token_f1']:.3f} ({dt:.1f}s)"
            )

        mean_exact = float(np.mean(fold_exact_f1s))
        std_exact = float(np.std(fold_exact_f1s))
        mean_relaxed = float(np.mean(fold_relaxed_f1s))
        std_relaxed = float(np.std(fold_relaxed_f1s))
        mean_tok = float(np.mean(fold_token_f1s))
        std_tok = float(np.std(fold_token_f1s))
        mean_diag = float(np.mean(fold_diag_f1s))
        std_diag = float(np.std(fold_diag_f1s))

        row = {
            "rank": idx,
            "algorithm": cfg["algorithm"],
            "c1": cfg["c1"],
            "c2": cfg["c2"],
            "min_freq": cfg["min_freq"],
            "all_possible_transitions": cfg["all_possible_transitions"],
            "max_iterations": cfg["max_iterations"],
            "stage1_dev_f1": cfg["stage1_dev_f1"],
            "cv_exact_f1_mean": round(mean_exact, 4),
            "cv_exact_f1_std": round(std_exact, 4),
            "cv_relaxed_f1_mean": round(mean_relaxed, 4),
            "cv_relaxed_f1_std": round(std_relaxed, 4),
            "cv_token_f1_mean": round(mean_tok, 4),
            "cv_token_f1_std": round(std_tok, 4),
            "cv_diag_rel_f1_mean": round(mean_diag, 4),
            "cv_diag_rel_f1_std": round(std_diag, 4),
            "fold_exact_f1s": str([round(x, 4) for x in fold_exact_f1s]),
        }
        results.append(row)

    results.sort(key=lambda r: r["cv_exact_f1_mean"], reverse=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved Stage 2 CV results to: {out_csv}")
    print("\n=== STAGE 2 CV RANKINGS (5-Fold Report-Grouped CV) ===")
    print(
        f"{'Rank':4} {'Algo':6} {'c1':6} {'c2':6} {'mf':3} {'trans':5} "
        f"{'CV Exact F1':18} {'CV Relax F1':18} {'CV Tok F1':18} {'Dev F1':8}"
    )
    for i, r in enumerate(results, 1):
        print(
            f"{i:4d} {r['algorithm']:6} {r['c1']:6.3f} {r['c2']:6.3f} {r['min_freq']:3d} {str(r['all_possible_transitions']):5} "
            f"{r['cv_exact_f1_mean']:.3f} +/- {r['cv_exact_f1_std']:.3f}   "
            f"{r['cv_relaxed_f1_mean']:.3f} +/- {r['cv_relaxed_f1_std']:.3f}   "
            f"{r['cv_token_f1_mean']:.3f} +/- {r['cv_token_f1_std']:.3f}   "
            f"{r['stage1_dev_f1']:8.3f}"
        )


if __name__ == "__main__":
    main()
