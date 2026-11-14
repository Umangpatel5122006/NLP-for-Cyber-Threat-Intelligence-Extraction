"""Evaluation harness for CTI extraction.

Provides multi-perspective metrics (exact seqeval, strict IOB2, relaxed span overlap,
token-level micro/macro), document-grouped cross-validation, cluster bootstrapping,
error taxonomy analysis, and per-source breakdown.
"""

from cti.eval.bootstrap import cluster_bootstrap_ci, bootstrap_suite
from cti.eval.cv import get_report_grouped_splits
from cti.eval.metrics import (
    classify_errors,
    evaluate_predictions,
    exact_span_metrics,
    extract_spans,
    relaxed_span_metrics,
    token_level_metrics,
)
from cti.eval.report import (
    export_report_csv,
    export_report_markdown,
    generate_evaluation_summary,
    per_source_breakdown,
)

__all__ = [
    "extract_spans",
    "exact_span_metrics",
    "relaxed_span_metrics",
    "token_level_metrics",
    "classify_errors",
    "evaluate_predictions",
    "get_report_grouped_splits",
    "cluster_bootstrap_ci",
    "bootstrap_suite",
    "per_source_breakdown",
    "generate_evaluation_summary",
    "export_report_markdown",
    "export_report_csv",
]
