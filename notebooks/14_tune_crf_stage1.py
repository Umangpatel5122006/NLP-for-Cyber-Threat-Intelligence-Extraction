"""Stage 1: Coarse random search over CRF hyperparameters.

Tests algorithms (lbfgs, l2sgd, ap, pa, arow), regularization (c1, c2 in [0.01, 3]),
min_freq in {0, 2, 3, 5}, all_possible_transitions in {True, False}, and max_iterations
(100 vs 300) with convergence tracking.
Uses precomputed/cached features (joblib) to keep search time fast.
Saves partial results to results/crf_tuning/stage1_coarse_search.csv after each draw.
"""

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import pycrfsuite
import sklearn_crfsuite

from cti.core.dataset import load_smah
from cti.eval.metrics import exact_span_metrics, relaxed_span_metrics, token_level_metrics
from cti.models.crf import _sentence_features, repair_bio_sequence
from cti.models.data import load_bio
from cti.models.feature_cache import get_cached_sentence_features

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "crf_tuning"
CACHE_DIR = RESULTS_DIR / "cache"


def log_uniform(low: float, high: float, rng: random.Random) -> float:
    """Sample from a log-uniform distribution between low and high."""
    log_low = math.log10(low)
    log_high = math.log10(high)
    return 10 ** rng.uniform(log_low, log_high)


def train_and_eval_crf(
    algorithm: str,
    c1: float,
    c2: float,
    min_freq: int,
    all_possible_transitions: bool,
    max_iterations: int,
    X_train: List[List[Dict[str, Any]]],
    y_train: List[List[str]],
    X_dev: List[List[Dict[str, Any]]],
    y_dev: List[List[str]],
    dev_rel_mask: List[bool],
) -> Dict[str, Any]:
    """Train a single CRF configuration and evaluate on dev split."""
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.select(algorithm)
    params = {
        "max_iterations": max_iterations,
        "feature.minfreq": min_freq,
        "feature.possible_transitions": 1 if all_possible_transitions else 0,
    }
    if algorithm == "lbfgs":
        params["c1"] = c1
        params["c2"] = c2
    elif algorithm == "l2sgd":
        params["c2"] = c2

    trainer.set_params(params)

    model_tmp = CACHE_DIR / f"temp_{random.randint(100000, 999999)}.crf"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    trainer.train(str(model_tmp))
    fit_time = time.perf_counter() - t0

    # Determine convergence
    actual_iterations = max_iterations
    converged = False
    if hasattr(trainer, "logparser") and trainer.logparser and hasattr(trainer.logparser, "last_iteration"):
        last_it = trainer.logparser.last_iteration
        if isinstance(last_it, dict) and "num" in last_it:
            actual_iterations = last_it["num"]
            converged = actual_iterations < max_iterations

    # Predict on dev
    tagger = pycrfsuite.Tagger()
    tagger.open(str(model_tmp))
    raw_preds = [tagger.tag(xseq) for xseq in X_dev]
    predictions = [repair_bio_sequence(seq) for seq in raw_preds]
    tagger.close()
    if model_tmp.exists():
        os.remove(model_tmp)

    # All-dev headline scores
    exact_all = exact_span_metrics(y_dev, predictions)
    relaxed_all = relaxed_span_metrics(y_dev, predictions)
    token_all = token_level_metrics(y_dev, predictions)

    # Relevant-dev diagnostic scores
    rel_gold = [g for g, k in zip(y_dev, dev_rel_mask) if k]
    rel_pred = [p for p, k in zip(predictions, dev_rel_mask) if k]
    exact_rel = exact_span_metrics(rel_gold, rel_pred)

    pl = exact_rel["per_label"]

    return {
        "algorithm": algorithm,
        "c1": round(c1, 4),
        "c2": round(c2, 4),
        "min_freq": min_freq,
        "all_possible_transitions": all_possible_transitions,
        "max_iterations": max_iterations,
        "actual_iterations": actual_iterations,
        "converged": converged,
        "fit_time_s": round(fit_time, 2),
        "all_dev_exact_f1": exact_all["f1"],
        "all_dev_relaxed_f1": relaxed_all["f1"],
        "all_dev_token_macro_f1": token_all["macro_f1"],
        "rel_dev_exact_f1": exact_rel["f1"],
        "threat_actor_f1": pl.get("THREAT_ACTOR", {}).get("f1", 0.0),
        "malware_tool_f1": pl.get("MALWARE_TOOL", {}).get("f1", 0.0),
        "infra_f1": pl.get("INFRASTRUCTURE", {}).get("f1", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", default="entities", choices=("entities", "action"))
    parser.add_argument("--n-draws", type=int, default=30, help="Number of random parameter draws.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"stage1_coarse_search_{args.layer}.csv"

    print(f"=== Stage 1 Coarse Search: Layer='{args.layer}', Draws={args.n_draws} ===")

    # 1. Load data
    raw_splits = load_smah()
    train_x, train_y = load_bio(args.layer, "train")
    dev_x, dev_y = load_bio(args.layer, "dev")

    raw_train = raw_splits["train"]
    train_rel_mask = [str(r["relevant"]).casefold() == "yes" for _, r in raw_train.iterrows()]
    rel_train_x = [s for s, k in zip(train_x, train_rel_mask) if k]
    rel_train_y = [s for s, k in zip(train_y, train_rel_mask) if k]

    raw_dev = raw_splits["dev"]
    dev_rel_mask = [str(r["relevant"]).casefold() == "yes" for _, r in raw_dev.iterrows()]

    # 2. Extract and cache features
    cache_train = CACHE_DIR / f"feats_train_rel_{args.layer}.joblib"
    cache_dev = CACHE_DIR / f"feats_dev_{args.layer}.joblib"

    print("Retrieving/extracting features...")
    X_train = get_cached_sentence_features(rel_train_x, cache_path=cache_train, verbose=True)
    X_dev = get_cached_sentence_features(dev_x, cache_path=cache_dev, verbose=True)
    print(f"Ready: {len(X_train)} train sentences, {len(X_dev)} dev sentences.")

    # 3. Candidate grid distributions
    algorithms = ["lbfgs", "l2sgd", "ap", "pa", "arow"]
    algo_weights = [0.45, 0.25, 0.10, 0.10, 0.10]
    min_freq_options = [0, 2, 3, 5]

    fieldnames = [
        "draw", "algorithm", "c1", "c2", "min_freq", "all_possible_transitions",
        "max_iterations", "actual_iterations", "converged", "fit_time_s",
        "all_dev_exact_f1", "all_dev_relaxed_f1", "all_dev_token_macro_f1",
        "rel_dev_exact_f1", "threat_actor_f1", "malware_tool_f1", "infra_f1"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    results = []

    # Always include baseline (draw 1) as reference
    configs = [{
        "algorithm": "lbfgs",
        "c1": 1.0,
        "c2": 1.0,
        "min_freq": 0,
        "all_possible_transitions": True,
        "max_iterations": 200,
    }]

    for _ in range(args.n_draws - 1):
        algo = rng.choices(algorithms, weights=algo_weights, k=1)[0]
        c1 = log_uniform(0.01, 3.0, rng) if algo == "lbfgs" else 0.0
        c2 = log_uniform(0.01, 3.0, rng) if algo in ("lbfgs", "l2sgd") else 0.0
        min_freq = rng.choice(min_freq_options)
        trans = rng.choice([True, False])
        max_iter = rng.choice([100, 300])
        configs.append({
            "algorithm": algo,
            "c1": c1,
            "c2": c2,
            "min_freq": min_freq,
            "all_possible_transitions": trans,
            "max_iterations": max_iter,
        })

    for draw_idx, cfg in enumerate(configs, 1):
        print(
            f"Draw {draw_idx:2d}/{len(configs)}: {cfg['algorithm']} "
            f"c1={cfg['c1']:.3f} c2={cfg['c2']:.3f} min_freq={cfg['min_freq']} "
            f"trans={cfg['all_possible_transitions']} max_iter={cfg['max_iterations']}...",
            end=" ",
            flush=True,
        )
        res = train_and_eval_crf(
            algorithm=cfg["algorithm"],
            c1=cfg["c1"],
            c2=cfg["c2"],
            min_freq=cfg["min_freq"],
            all_possible_transitions=cfg["all_possible_transitions"],
            max_iterations=cfg["max_iterations"],
            X_train=X_train,
            y_train=rel_train_y,
            X_dev=X_dev,
            y_dev=dev_y,
            dev_rel_mask=dev_rel_mask,
        )
        row = {"draw": draw_idx, **res}
        results.append(row)

        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        print(
            f"Done in {res['fit_time_s']}s (it={res['actual_iterations']}/{cfg['max_iterations']}) | "
            f"All-dev F1={res['all_dev_exact_f1']:.3f} Rel-dev F1={res['rel_dev_exact_f1']:.3f}"
        )

    # Rank and print top 5
    results.sort(key=lambda r: -r["all_dev_exact_f1"])
    print("\n=== TOP 5 CONFIGURATIONS (by Headline All-Dev Exact F1) ===")
    print(
        f"{'Rank':4} {'Draw':4} {'Algo':6} {'c1':6} {'c2':6} {'mf':3} {'trans':5} "
        f"{'iter':8} {'Conv':5} {'All F1':8} {'Rel F1':8} {'Relax':8} {'Tok F1':8}"
    )
    for rank, r in enumerate(results[:5], 1):
        print(
            f"{rank:4d} {r['draw']:4d} {r['algorithm']:6} {r['c1']:6.3f} {r['c2']:6.3f} "
            f"{r['min_freq']:3d} {str(r['all_possible_transitions']):5} "
            f"{r['actual_iterations']:4d}/{r['max_iterations']:<3d} {str(r['converged']):5} "
            f"{r['all_dev_exact_f1']:8.3f} {r['rel_dev_exact_f1']:8.3f} "
            f"{r['all_dev_relaxed_f1']:8.3f} {r['all_dev_token_macro_f1']:8.3f}"
        )


if __name__ == "__main__":
    main()
