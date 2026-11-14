"""Reporting utilities for CTI sequence tagging evaluation.

Produces structured tables, per-source breakdowns, error taxonomies, and exports to
Markdown and CSV in the results/ directory.
Strictly distinguishes Headline (all-sentence end-to-end) from Diagnostic (relevant-only).
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from cti.eval.metrics import evaluate_predictions, exact_span_metrics, relaxed_span_metrics, token_level_metrics


def per_source_breakdown(
    gold: List[List[str]],
    predictions: List[List[str]],
    sources: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute exact, relaxed, and token-level metrics sliced by document source."""
    if not (len(gold) == len(predictions) == len(sources)):
        raise ValueError("gold, predictions, and sources must have identical lengths.")

    by_source_gold: Dict[str, List[List[str]]] = defaultdict(list)
    by_source_pred: Dict[str, List[List[str]]] = defaultdict(list)

    for g, p, s in zip(gold, predictions, sources):
        by_source_gold[s].append(g)
        by_source_pred[s].append(p)

    results = {}
    for src in sorted(by_source_gold):
        s_gold = by_source_gold[src]
        s_pred = by_source_pred[src]
        exact = exact_span_metrics(s_gold, s_pred)
        relaxed = relaxed_span_metrics(s_gold, s_pred)
        token = token_level_metrics(s_gold, s_pred)

        results[src] = {
            "sentence_count": len(s_gold),
            "exact_p": exact["precision"],
            "exact_r": exact["recall"],
            "exact_f1": exact["f1"],
            "relaxed_p": relaxed["precision"],
            "relaxed_r": relaxed["recall"],
            "relaxed_f1": relaxed["f1"],
            "token_macro_f1": token["macro_f1"],
            "per_label": exact["per_label"],
        }
    return results


def generate_evaluation_summary(
    model_name: str,
    layer: str,
    gold: List[List[str]],
    predictions: List[List[str]],
    doc_ids: List[str],
    sources: List[str],
    relevant_mask: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """Compile a complete evaluation report bundle with headline and diagnostic metrics."""
    # 1. Headline (all sentences end-to-end)
    headline_eval = evaluate_predictions(gold, predictions)
    source_results = per_source_breakdown(gold, predictions, sources)

    summary = {
        "model_name": model_name,
        "layer": layer,
        "total_sentences": len(gold),
        "headline_all_sentences": headline_eval,
        "per_source": source_results,
    }

    # 2. Diagnostic (relevant-only)
    if relevant_mask is not None:
        rel_gold = [g for g, keep in zip(gold, relevant_mask) if keep]
        rel_pred = [p for p, keep in zip(predictions, relevant_mask) if keep]
        rel_eval = evaluate_predictions(rel_gold, rel_pred)
        summary["diagnostic_relevant_only"] = {
            "relevant_sentences": len(rel_gold),
            "evaluation": rel_eval,
        }

    return summary


def export_report_markdown(summary: Dict[str, Any], output_path: Path):
    """Write an evaluation report formatted as a Markdown document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m = summary["model_name"]
    layer = summary["layer"]
    h = summary["headline_all_sentences"]
    h_exact = h["exact_default"]
    h_rel = h["relaxed"]
    h_tok = h["token_level"]
    err = h["error_taxonomy"]["counts"]

    lines = [
        f"# Evaluation Report: {m} (Layer: {layer})",
        "",
        "## 1. Headline Metrics (All-Sentence End-to-End)",
        f"- **Total Sentences Evaluated**: {summary['total_sentences']}",
        "",
        "| Metric Mode | Precision | Recall | F1 Score |",
        "|---|---:|---:|---:|",
        f"| Exact-Span (seqeval default) | {h_exact['precision']:.3f} | {h_exact['recall']:.3f} | **{h_exact['f1']:.3f}** |",
        f"| Exact-Span (strict IOB2) | {h['exact_strict']['precision']:.3f} | {h['exact_strict']['recall']:.3f} | {h['exact_strict']['f1']:.3f} |",
        f"| Relaxed-Span (Overlap) | {h_rel['precision']:.3f} | {h_rel['recall']:.3f} | {h_rel['f1']:.3f} |",
        f"| Token-Level (Macro) | {h_tok['macro_precision']:.3f} | {h_tok['macro_recall']:.3f} | {h_tok['macro_f1']:.3f} |",
        "",
        "### Per-Label Exact-Span Performance",
        "| Entity Label | Precision | Recall | F1 Score | Gold Support | Predicted Spans |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for lbl, stats in h_exact["per_label"].items():
        lines.append(
            f"| `{lbl}` | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1']:.3f} | {stats['support']} | {stats['predicted']} |"
        )

    if "diagnostic_relevant_only" in summary:
        d = summary["diagnostic_relevant_only"]
        d_eval = d["evaluation"]
        d_exact = d_eval["exact_default"]
        d_rel = d_eval["relaxed"]
        lines.extend([
            "",
            "## 2. Diagnostic Metrics (Relevant Sentences Only)",
            f"- **Relevant Sentences**: {d['relevant_sentences']} / {summary['total_sentences']}",
            "> [!NOTE]",
            "> Relevant-only metrics evaluate model precision/recall solely on ground-truth relevant sentences.",
            "> They isolate tagger capabilities from sentence relevance filtering errors.",
            "",
            "| Metric Mode | Precision | Recall | F1 Score |",
            "|---|---:|---:|---:|",
            f"| Exact-Span (seqeval default) | {d_exact['precision']:.3f} | {d_exact['recall']:.3f} | **{d_exact['f1']:.3f}** |",
            f"| Relaxed-Span (Overlap) | {d_rel['precision']:.3f} | {d_rel['recall']:.3f} | {d_rel['f1']:.3f} |",
            f"| Token-Level (Macro) | {d_eval['token_level']['macro_precision']:.3f} | {d_eval['token_level']['macro_recall']:.3f} | {d_eval['token_level']['macro_f1']:.3f} |",
            "",
            "### Per-Label Diagnostic (Relevant-Only)",
            "| Entity Label | Precision | Recall | F1 Score | Gold Support | Predicted Spans |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for lbl, stats in d_exact["per_label"].items():
            lines.append(
                f"| `{lbl}` | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1']:.3f} | {stats['support']} | {stats['predicted']} |"
            )

    lines.extend([
        "",
        "## 3. Error Taxonomy Breakdown (All Sentences)",
        "| Error Category | Count | Percentage of Gold/Pred | Description |",
        "|---|---:|---:|---|",
        f"| Exact Match | {err['exact_match']} | - | Identical label and token boundaries |",
        f"| Boundary Error | {err['boundary_error']} | - | Matching label & overlapping span, but boundary mismatch |",
        f"| Wrong Type (Confusion) | {err['wrong_type']} | - | Overlapping span, but conflicting entity labels |",
        f"| Missed (False Negative) | {err['missed']} | - | Gold entity completely missed by model |",
        f"| Spurious (False Positive) | {err['spurious']} | - | Predicted entity with no overlapping gold span |",
        "",
        "## 4. Per-Source Breakdown",
        "| Source | Sentences | Exact P | Exact R | Exact F1 | Relaxed F1 | Token Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for src, s in summary["per_source"].items():
        lines.append(
            f"| `{src}` | {s['sentence_count']} | {s['exact_p']:.3f} | {s['exact_r']:.3f} | {s['exact_f1']:.3f} | {s['relaxed_f1']:.3f} | {s['token_macro_f1']:.3f} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_report_csv(summary: Dict[str, Any], output_path: Path):
    """Write an evaluation report formatted as a flat CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    m = summary["model_name"]
    layer = summary["layer"]
    h_exact = summary["headline_all_sentences"]["exact_default"]
    h_rel = summary["headline_all_sentences"]["relaxed"]
    h_tok = summary["headline_all_sentences"]["token_level"]

    # Headline overall
    rows.append({
        "model": m,
        "layer": layer,
        "scope": "headline_all_dev",
        "partition": "all",
        "metric_mode": "exact_span_default",
        "precision": h_exact["precision"],
        "recall": h_exact["recall"],
        "f1": h_exact["f1"],
    })
    rows.append({
        "model": m,
        "layer": layer,
        "scope": "headline_all_dev",
        "partition": "all",
        "metric_mode": "relaxed_span",
        "precision": h_rel["precision"],
        "recall": h_rel["recall"],
        "f1": h_rel["f1"],
    })
    rows.append({
        "model": m,
        "layer": layer,
        "scope": "headline_all_dev",
        "partition": "all",
        "metric_mode": "token_macro",
        "precision": h_tok["macro_precision"],
        "recall": h_tok["macro_recall"],
        "f1": h_tok["macro_f1"],
    })

    # Per-label headline
    for lbl, stats in h_exact["per_label"].items():
        rows.append({
            "model": m,
            "layer": layer,
            "scope": "headline_all_dev",
            "partition": lbl,
            "metric_mode": "exact_span_per_label",
            "precision": stats["precision"],
            "recall": stats["recall"],
            "f1": stats["f1"],
        })

    # Diagnostic relevant-only
    if "diagnostic_relevant_only" in summary:
        d_exact = summary["diagnostic_relevant_only"]["evaluation"]["exact_default"]
        d_rel = summary["diagnostic_relevant_only"]["evaluation"]["relaxed"]
        rows.append({
            "model": m,
            "layer": layer,
            "scope": "diagnostic_relevant_dev",
            "partition": "all",
            "metric_mode": "exact_span_default",
            "precision": d_exact["precision"],
            "recall": d_exact["recall"],
            "f1": d_exact["f1"],
        })
        rows.append({
            "model": m,
            "layer": layer,
            "scope": "diagnostic_relevant_dev",
            "partition": "all",
            "metric_mode": "relaxed_span",
            "precision": d_rel["precision"],
            "recall": d_rel["recall"],
            "f1": d_rel["f1"],
        })
        for lbl, stats in d_exact["per_label"].items():
            rows.append({
                "model": m,
                "layer": layer,
                "scope": "diagnostic_relevant_dev",
                "partition": lbl,
                "metric_mode": "exact_span_per_label",
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1"],
            })

    # Per-source
    for src, s in summary["per_source"].items():
        rows.append({
            "model": m,
            "layer": layer,
            "scope": "per_source",
            "partition": src,
            "metric_mode": "exact_span_default",
            "precision": s["exact_p"],
            "recall": s["exact_r"],
            "f1": s["exact_f1"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
