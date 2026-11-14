"""Stage 4: New feature exploration for CRF.

Explores two domain-informed classical feature augmentations using 5-fold
report-grouped cross-validation across all 80 train + dev reports:
  1. Document consistency: Tracks per-report token frequency and whether a
     token appeared earlier in the document (Krishnan & Manning, 2006 style).
  2. Unsupervised word clusters: Distributional clustering via context co-occurrence
     + TruncatedSVD + MiniBatchKMeans (Brown-style clusters, 50 clusters).
  3. Combined: Document consistency + word clusters.

Saves results to results/crf_tuning/stage4_new_features_{layer}.csv.
"""

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pycrfsuite
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer

from cti.core.dataset import load_smah
from cti.eval.cv import get_report_grouped_splits
from cti.eval.metrics import exact_span_metrics, relaxed_span_metrics, token_level_metrics
from cti.models.crf import _sentence_features, repair_bio_sequence
from cti.models.data import load_bio

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "crf_tuning"
CACHE_DIR = RESULTS_DIR / "cache"


def build_word_clusters(
    sentences: List[List[str]],
    n_clusters: int = 50,
    min_word_count: int = 3,
    window: int = 2,
    seed: int = 42,
) -> Dict[str, int]:
    """Build unsupervised distributional word clusters using context co-occurrence."""
    # Count word frequencies
    freqs = Counter(w.lower() for sent in sentences for w in sent)
    vocab = {w for w, c in freqs.items() if c >= min_word_count and len(w) > 1}
    word_to_idx = {w: i for i, w in enumerate(sorted(vocab))}
    
    if not word_to_idx:
        return {}

    # Build context dictionaries for each target word
    contexts = defaultdict(Counter)
    for sent in sentences:
        lowers = [w.lower() for w in sent]
        n = len(lowers)
        for i, target in enumerate(lowers):
            if target in word_to_idx:
                for d in range(-window, window + 1):
                    if d != 0 and 0 <= i + d < n:
                        ctx_w = lowers[i + d]
                        contexts[target][f"{d}:{ctx_w}"] += 1

    target_words = sorted(word_to_idx.keys())
    dict_list = [contexts[w] for w in target_words]

    vec = DictVectorizer(sparse=True)
    X_ctx = vec.fit_transform(dict_list)

    n_components = min(30, X_ctx.shape[1] - 1, X_ctx.shape[0] - 1)
    if n_components > 2:
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        X_reduced = svd.fit_transform(X_ctx)
    else:
        X_reduced = X_ctx.toarray()

    k = min(n_clusters, len(target_words))
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=256)
    clusters = kmeans.fit_predict(X_reduced)

    return {w: int(c) for w, c in zip(target_words, clusters)}


def build_doc_contexts(
    sentences: List[List[str]],
    doc_ids: List[str],
) -> List[Dict[str, Dict[str, Any]]]:
    """Build document context dictionaries for each sentence."""
    # Group sentences by doc_id
    doc_groups = defaultdict(list)
    for idx, (doc_id, sent) in enumerate(zip(doc_ids, sentences)):
        doc_groups[doc_id].append((idx, sent))

    doc_contexts = [{} for _ in sentences]

    for doc_id, items in doc_groups.items():
        doc_freqs = Counter(w.lower() for _, sent in items for w in sent)
        seen_words = set()

        for idx, sent in items:
            sent_context = {}
            for w in sent:
                lower = w.lower()
                sent_context[lower] = {
                    "freq": doc_freqs[lower],
                    "seen_earlier": lower in seen_words,
                }
            # Update seen words after this sentence
            for w in sent:
                seen_words.add(w.lower())
            doc_contexts[idx] = sent_context

    return doc_contexts


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

    algo = config.get("algorithm", "lbfgs")
    trainer.select(algo)
    params = {
        "max_iterations": config.get("max_iterations", 100),
        "feature.minfreq": config.get("min_freq", 0),
        "feature.possible_transitions": 1 if config.get("all_possible_transitions", True) else 0,
    }
    if algo == "lbfgs":
        params["c1"] = config.get("c1", 1.0)
        params["c2"] = config.get("c2", 1.0)
    elif algo == "l2sgd":
        params["c2"] = config.get("c2", 1.0)
    trainer.set_params(params)

    tmp_path = CACHE_DIR / f"newfeat_temp_{time.time_ns()}.crf"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    trainer.train(str(tmp_path))

    tagger = pycrfsuite.Tagger()
    tagger.open(str(tmp_path))
    raw_preds = [tagger.tag(xseq) for xseq in val_x_features]
    preds = [repair_bio_sequence(seq) for seq in raw_preds]
    tagger.close()
    if tmp_path.exists():
        tmp_path.unlink()

    exact_all = exact_span_metrics(val_y, preds)
    relaxed_all = relaxed_span_metrics(val_y, preds)
    tok_all = token_level_metrics(val_y, preds)

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
    parser.add_argument("--algorithm", default="lbfgs")
    parser.add_argument("--c1", type=float, default=1.0)
    parser.add_argument("--c2", type=float, default=1.0)
    parser.add_argument("--min-freq", type=int, default=0)
    parser.add_argument("--all-possible-transitions", action="store_true", default=True)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_config = {
        "algorithm": args.algorithm,
        "c1": args.c1,
        "c2": args.c2,
        "min_freq": args.min_freq,
        "all_possible_transitions": args.all_possible_transitions,
        "max_iterations": args.max_iterations,
    }

    print(f"=== Stage 4: New Feature Exploration ({args.layer}) ===")
    print(f"CRF Params: {model_config}")

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

    splits = get_report_grouped_splits(all_doc_ids, n_splits=args.n_splits, seed=args.seed)

    print("Computing document-level contexts...")
    doc_contexts = build_doc_contexts(all_x, all_doc_ids)

    print("Building unsupervised word clusters (50 clusters)...")
    t0_clust = time.perf_counter()
    word_clusters = build_word_clusters(all_x, n_clusters=50, seed=args.seed)
    print(f"Clustered {len(word_clusters)} vocabulary items in {time.perf_counter() - t0_clust:.2f}s.")

    conditions = [
        ("baseline", None, None),
        ("+doc_consistency", doc_contexts, None),
        ("+word_clusters", None, word_clusters),
        ("+doc_and_clusters", doc_contexts, word_clusters),
    ]

    out_csv = RESULTS_DIR / f"stage4_new_features_{args.layer}.csv"
    fieldnames = [
        "condition",
        "cv_exact_f1_mean",
        "cv_exact_f1_std",
        "delta_exact_f1",
        "cv_relaxed_f1_mean",
        "cv_relaxed_f1_std",
        "cv_token_f1_mean",
        "cv_token_f1_std",
        "cv_diag_rel_f1_mean",
        "cv_diag_rel_f1_std",
        "decision",
        "elapsed_s",
    ]

    results = []
    baseline_exact_mean = None

    for cond_name, d_ctx, w_clust in conditions:
        print(f"\nCondition: {cond_name}...")
        t0 = time.perf_counter()

        # Extract features with custom doc_context or word_clusters
        feats = [
            _sentence_features(
                words,
                doc_context=(d_ctx[i] if d_ctx is not None else None),
                word_clusters=w_clust,
            )
            for i, words in enumerate(all_x)
        ]

        fold_exact_f1s = []
        fold_relaxed_f1s = []
        fold_token_f1s = []
        fold_diag_f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
            rel_train_idx = [i for i in train_idx if all_rel[i]]
            X_tr = [feats[i] for i in rel_train_idx]
            y_tr = [all_y[i] for i in rel_train_idx]

            X_val = [feats[i] for i in val_idx]
            y_val = [all_y[i] for i in val_idx]
            val_rel = [all_rel[i] for i in val_idx]

            f_res = run_fold(model_config, X_tr, y_tr, X_val, y_val, val_rel)
            fold_exact_f1s.append(f_res["headline_exact_f1"])
            fold_relaxed_f1s.append(f_res["headline_relaxed_f1"])
            fold_token_f1s.append(f_res["headline_token_f1"])
            fold_diag_f1s.append(f_res["diagnostic_exact_f1"])

        elapsed = time.perf_counter() - t0
        mean_exact = float(np.mean(fold_exact_f1s))
        std_exact = float(np.std(fold_exact_f1s))
        mean_relaxed = float(np.mean(fold_relaxed_f1s))
        std_relaxed = float(np.std(fold_relaxed_f1s))
        mean_tok = float(np.mean(fold_token_f1s))
        std_tok = float(np.std(fold_token_f1s))
        mean_diag = float(np.mean(fold_diag_f1s))
        std_diag = float(np.std(fold_diag_f1s))

        if cond_name == "baseline":
            baseline_exact_mean = mean_exact
            delta = 0.0
            decision = "BASELINE"
        else:
            delta = mean_exact - baseline_exact_mean
            if delta > 0.005:
                decision = "ADOPT (Clear improvement > 0.5% F1)"
            elif delta < -0.005:
                decision = "REJECT (Hurts performance)"
            else:
                decision = "NEUTRAL (Within noise)"

        row = {
            "condition": cond_name,
            "cv_exact_f1_mean": round(mean_exact, 4),
            "cv_exact_f1_std": round(std_exact, 4),
            "delta_exact_f1": round(delta, 4),
            "cv_relaxed_f1_mean": round(mean_relaxed, 4),
            "cv_relaxed_f1_std": round(std_relaxed, 4),
            "cv_token_f1_mean": round(mean_tok, 4),
            "cv_token_f1_std": round(std_tok, 4),
            "cv_diag_rel_f1_mean": round(mean_diag, 4),
            "cv_diag_rel_f1_std": round(std_diag, 4),
            "decision": decision,
            "elapsed_s": round(elapsed, 1),
        }
        results.append(row)
        print(
            f"  Exact F1: {mean_exact:.4f} +/- {std_exact:.4f} (delta={delta:+.4f}) -> {decision} [{elapsed:.1f}s]"
        )

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nSaved Stage 4 New Features results to: {out_csv}")
    print("\n=== STAGE 4 NEW FEATURES EXPLORATION SUMMARY ===")
    print(
        f"{'Condition':20} {'CV Exact F1':18} {'Delta':8} {'CV Relax F1':18} {'Decision'}"
    )
    for r in results:
        print(
            f"{r['condition']:20} {r['cv_exact_f1_mean']:.4f} +/- {r['cv_exact_f1_std']:.4f} "
            f"{r['delta_exact_f1']:+8.4f} "
            f"{r['cv_relaxed_f1_mean']:.4f} +/- {r['cv_relaxed_f1_std']:.4f} "
            f"{r['decision']}"
        )


if __name__ == "__main__":
    main()
