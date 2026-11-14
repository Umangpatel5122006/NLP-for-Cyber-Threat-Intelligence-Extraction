"""Evaluation harness for CTI extraction.

Provides multi-perspective metrics (exact seqeval, strict IOB2, relaxed span overlap,
token-level micro/macro), document-grouped cross-validation, cluster bootstrapping,
error taxonomy analysis, and per-source breakdown.
"""

from cti.eval.metrics import (
    classify_errors,
    evaluate_predictions,
    exact_span_metrics,
    extract_spans,
    relaxed_span_metrics,
    token_level_metrics,
)

__all__ = [
    "extract_spans",
    "exact_span_metrics",
    "relaxed_span_metrics",
    "token_level_metrics",
    "classify_errors",
    "evaluate_predictions",
]
