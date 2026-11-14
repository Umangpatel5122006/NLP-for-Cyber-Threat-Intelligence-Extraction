"""Cluster bootstrap confidence intervals for CTI evaluation.

Resamples whole intelligence reports (doc_id clusters) with replacement rather than
individual sentences, accounting for intra-document correlation and producing honest
empirical confidence intervals.
"""

import random
from typing import Callable, Dict, List, Tuple
import numpy as np

from cti.eval.metrics import exact_span_metrics, relaxed_span_metrics, token_level_metrics


def cluster_bootstrap_ci(
    gold: List[List[str]],
    predictions: List[List[str]],
    doc_ids: List[str],
    metric_fn: Callable[[List[List[str]], List[List[str]]], float],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute empirical percentile bootstrap confidence intervals by resampling reports.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    if len(gold) != len(predictions) or len(gold) != len(doc_ids):
        raise ValueError("gold, predictions, and doc_ids must have identical lengths.")

    unique_docs = sorted(list(set(doc_ids)))
    doc_to_indices: Dict[str, List[int]] = {}
    for idx, d in enumerate(doc_ids):
        doc_to_indices.setdefault(d, []).append(idx)

    point_estimate = metric_fn(gold, predictions)

    rng = random.Random(seed)
    boot_scores = []

    for _ in range(n_bootstraps):
        sampled_docs = rng.choices(unique_docs, k=len(unique_docs))
        sampled_indices = [
            idx for d in sampled_docs for idx in doc_to_indices[d]
        ]
        sub_gold = [gold[i] for i in sampled_indices]
        sub_pred = [predictions[i] for i in sampled_indices]

        score = metric_fn(sub_gold, sub_pred)
        boot_scores.append(score)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_scores, lower_pct))
    ci_upper = float(np.percentile(boot_scores, upper_pct))

    return round(point_estimate, 4), round(ci_lower, 4), round(ci_upper, 4)


def bootstrap_suite(
    gold: List[List[str]],
    predictions: List[List[str]],
    doc_ids: List[str],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute 95% bootstrap confidence intervals for exact, relaxed, and token-level F1."""
    def exact_f1(g, p):
        return exact_span_metrics(g, p)["f1"]

    def relaxed_f1(g, p):
        return relaxed_span_metrics(g, p)["f1"]

    def token_macro_f1(g, p):
        return token_level_metrics(g, p)["macro_f1"]

    metrics_map = {
        "exact_f1": exact_f1,
        "relaxed_f1": relaxed_f1,
        "token_macro_f1": token_macro_f1,
    }

    results = {}
    for name, fn in metrics_map.items():
        pe, lo, hi = cluster_bootstrap_ci(
            gold=gold,
            predictions=predictions,
            doc_ids=doc_ids,
            metric_fn=fn,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=seed,
        )
        results[name] = {"point": pe, "ci_lower": lo, "ci_upper": hi}

    return results
