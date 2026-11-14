"""Core evaluation metrics for sequence tagging in CTI reports.

Supports:
- Exact-span evaluation via seqeval (both default and strict IOB2 modes)
- Relaxed span evaluation (span overlap with matching label)
- Token-level micro and macro F1
- Error taxonomy: Exact Match, Boundary Error, Wrong Type, Missed, Spurious
"""

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from seqeval.metrics import (
    classification_report as seqeval_classification_report,
    f1_score as seqeval_f1_score,
    precision_score as seqeval_precision_score,
    recall_score as seqeval_recall_score,
)
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.scheme import IOB2
from sklearn.metrics import precision_recall_fscore_support


def extract_spans(sequence: List[str]) -> List[Tuple[str, int, int]]:
    """Extract entity spans as (label, start_token, end_token) with end exclusive.
    
    Uses seqeval get_entities to guarantee consistency with exact span scoring.
    """
    return [(label, start, end + 1) for label, start, end in get_entities(sequence)]


def exact_span_metrics(
    gold: List[List[str]],
    predictions: List[List[str]],
    mode: str = "default",
) -> Dict[str, Any]:
    """Calculate exact-span precision, recall, and F1 per label and overall.
    
    Parameters:
        gold: list of true BIO tag sequences
        predictions: list of predicted BIO tag sequences
        mode: 'default' for standard seqeval, or 'strict' for strict IOB2
    """
    if len(gold) != len(predictions):
        raise ValueError(
            f"Sequence count mismatch: {len(gold)} gold vs {len(predictions)} predictions."
        )

    kwargs = {}
    if mode == "strict":
        kwargs = {"mode": "strict", "scheme": IOB2}

    overall_p = float(seqeval_precision_score(gold, predictions, zero_division=0, **kwargs))
    overall_r = float(seqeval_recall_score(gold, predictions, zero_division=0, **kwargs))
    overall_f1 = float(seqeval_f1_score(gold, predictions, zero_division=0, **kwargs))

    # Per-label extraction
    gold_by_label: Dict[str, int] = defaultdict(int)
    pred_by_label: Dict[str, int] = defaultdict(int)
    tp_by_label: Dict[str, int] = defaultdict(int)

    all_labels: Set[str] = set()
    for g_seq, p_seq in zip(gold, predictions):
        g_spans = set(extract_spans(g_seq))
        p_spans = set(extract_spans(p_seq))

        for lbl, _, _ in g_spans:
            gold_by_label[lbl] += 1
            all_labels.add(lbl)
        for lbl, _, _ in p_spans:
            pred_by_label[lbl] += 1
            all_labels.add(lbl)
        for span in g_spans & p_spans:
            tp_by_label[span[0]] += 1

    per_label = {}
    for lbl in sorted(all_labels):
        tp = tp_by_label[lbl]
        pred_cnt = pred_by_label[lbl]
        gold_cnt = gold_by_label[lbl]
        p = tp / pred_cnt if pred_cnt > 0 else 0.0
        r = tp / gold_cnt if gold_cnt > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_label[lbl] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": gold_cnt,
            "predicted": pred_cnt,
            "true_positives": tp,
        }

    return {
        "mode": mode,
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "f1": round(overall_f1, 4),
        "per_label": per_label,
    }


def relaxed_span_metrics(
    gold: List[List[str]],
    predictions: List[List[str]],
) -> Dict[str, Any]:
    """Calculate relaxed-span metrics where spans with matching labels overlap.
    
    A predicted span is a true positive if it overlaps at least one gold span of the same label.
    A gold span is recalled if at least one predicted span of the same label overlaps it.
    """
    if len(gold) != len(predictions):
        raise ValueError("Sequence count mismatch.")

    all_labels: Set[str] = set()
    total_gold: Dict[str, int] = defaultdict(int)
    total_pred: Dict[str, int] = defaultdict(int)
    recalled_gold: Dict[str, int] = defaultdict(int)
    valid_pred: Dict[str, int] = defaultdict(int)

    for g_seq, p_seq in zip(gold, predictions):
        g_spans = extract_spans(g_seq)
        p_spans = extract_spans(p_seq)

        for lbl, _, _ in g_spans:
            total_gold[lbl] += 1
            all_labels.add(lbl)
        for lbl, _, _ in p_spans:
            total_pred[lbl] += 1
            all_labels.add(lbl)

        # Check gold spans for matching overlapping prediction
        for g_lbl, g_start, g_end in g_spans:
            for p_lbl, p_start, p_end in p_spans:
                if g_lbl == p_lbl and max(g_start, p_start) < min(g_end, p_end):
                    recalled_gold[g_lbl] += 1
                    break

        # Check predicted spans for matching overlapping gold
        for p_lbl, p_start, p_end in p_spans:
            for g_lbl, g_start, g_end in g_spans:
                if g_lbl == p_lbl and max(g_start, p_start) < min(g_end, p_end):
                    valid_pred[p_lbl] += 1
                    break

    per_label = {}
    total_tp_p = sum(valid_pred.values())
    total_pred_cnt = sum(total_pred.values())
    total_tp_g = sum(recalled_gold.values())
    total_gold_cnt = sum(total_gold.values())

    for lbl in sorted(all_labels):
        tp_p = valid_pred[lbl]
        pred_cnt = total_pred[lbl]
        tp_g = recalled_gold[lbl]
        gold_cnt = total_gold[lbl]

        p = tp_p / pred_cnt if pred_cnt > 0 else 0.0
        r = tp_g / gold_cnt if gold_cnt > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_label[lbl] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": gold_cnt,
            "predicted": pred_cnt,
        }

    overall_p = total_tp_p / total_pred_cnt if total_pred_cnt > 0 else 0.0
    overall_r = total_tp_g / total_gold_cnt if total_gold_cnt > 0 else 0.0
    overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0

    return {
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "f1": round(overall_f1, 4),
        "per_label": per_label,
    }


def token_level_metrics(
    gold: List[List[str]],
    predictions: List[List[str]],
) -> Dict[str, Any]:
    """Calculate token-level precision, recall, and F1 across entity types."""
    if len(gold) != len(predictions):
        raise ValueError("Sequence count mismatch.")

    # Flatten tokens and strip B-/I- prefixes to get raw entity types (or 'O')
    flat_gold = [
        tag.split("-", 1)[1] if "-" in tag else "O"
        for seq in gold for tag in seq
    ]
    flat_pred = [
        tag.split("-", 1)[1] if "-" in tag else "O"
        for seq in predictions for tag in seq
    ]

    labels = sorted(list(set(flat_gold) | set(flat_pred) - {"O"}))
    if not labels:
        return {
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_label": {},
        }

    p_per, r_per, f_per, s_per = precision_recall_fscore_support(
        flat_gold, flat_pred, labels=labels, zero_division=0
    )

    per_label = {}
    for lbl, p, r, f, s in zip(labels, p_per, r_per, f_per, s_per):
        per_label[lbl] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }

    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        flat_gold, flat_pred, labels=labels, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        flat_gold, flat_pred, labels=labels, average="macro", zero_division=0
    )

    return {
        "micro_precision": round(float(micro_p), 4),
        "micro_recall": round(float(micro_r), 4),
        "micro_f1": round(float(micro_f), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f), 4),
        "per_label": per_label,
    }


def classify_errors(
    gold: List[List[str]],
    predictions: List[List[str]],
) -> Dict[str, Any]:
    """Classify model predictions into an error taxonomy.
    
    Categories:
    - exact_match: same label, identical token boundary
    - boundary_error: same label, overlapping span, non-identical boundary
    - wrong_type: overlapping span, different label
    - missed: gold span with zero overlapping predicted spans
    - spurious: predicted span with zero overlapping gold spans
    """
    counts = {
        "exact_match": 0,
        "boundary_error": 0,
        "wrong_type": 0,
        "missed": 0,
        "spurious": 0,
    }
    per_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    confusion_pairs: Dict[Tuple[str, str], int] = defaultdict(int)

    for g_seq, p_seq in zip(gold, predictions):
        g_spans = extract_spans(g_seq)
        p_spans = extract_spans(p_seq)

        matched_p_indices: Set[int] = set()
        matched_g_indices: Set[int] = set()

        # Check for exact matches first
        for g_idx, g in enumerate(g_spans):
            for p_idx, p in enumerate(p_spans):
                if p_idx in matched_p_indices:
                    continue
                if g == p:
                    counts["exact_match"] += 1
                    per_label_counts[g[0]]["exact_match"] += 1
                    matched_g_indices.add(g_idx)
                    matched_p_indices.add(p_idx)
                    break

        # Check remaining for boundary errors (same label, overlapping)
        for g_idx, g in enumerate(g_spans):
            if g_idx in matched_g_indices:
                continue
            for p_idx, p in enumerate(p_spans):
                if p_idx in matched_p_indices:
                    continue
                if g[0] == p[0] and max(g[1], p[1]) < min(g[2], p[2]):
                    counts["boundary_error"] += 1
                    per_label_counts[g[0]]["boundary_error"] += 1
                    matched_g_indices.add(g_idx)
                    matched_p_indices.add(p_idx)
                    break

        # Check remaining for wrong type (different label, overlapping)
        for g_idx, g in enumerate(g_spans):
            if g_idx in matched_g_indices:
                continue
            for p_idx, p in enumerate(p_spans):
                if p_idx in matched_p_indices:
                    continue
                if g[0] != p[0] and max(g[1], p[1]) < min(g[2], p[2]):
                    counts["wrong_type"] += 1
                    per_label_counts[g[0]]["wrong_type"] += 1
                    confusion_pairs[(g[0], p[0])] += 1
                    matched_g_indices.add(g_idx)
                    matched_p_indices.add(p_idx)
                    break

        # Unmatched gold spans are missed
        for g_idx, g in enumerate(g_spans):
            if g_idx not in matched_g_indices:
                counts["missed"] += 1
                per_label_counts[g[0]]["missed"] += 1

        # Unmatched predicted spans are spurious
        for p_idx, p in enumerate(p_spans):
            if p_idx not in matched_p_indices:
                counts["spurious"] += 1
                per_label_counts[p[0]]["spurious"] += 1

    total_gold_spans = counts["exact_match"] + counts["boundary_error"] + counts["wrong_type"] + counts["missed"]
    total_pred_spans = counts["exact_match"] + counts["boundary_error"] + counts["wrong_type"] + counts["spurious"]

    return {
        "counts": counts,
        "total_gold": total_gold_spans,
        "total_predicted": total_pred_spans,
        "per_label": {lbl: dict(c) for lbl, c in per_label_counts.items()},
        "type_confusion_pairs": {
            f"{g_lbl} -> {p_lbl}": cnt
            for (g_lbl, p_lbl), cnt in sorted(confusion_pairs.items(), key=lambda x: -x[1])
        },
    }


def evaluate_predictions(
    gold: List[List[str]],
    predictions: List[List[str]],
) -> Dict[str, Any]:
    """Execute complete multi-perspective evaluation on prediction sequences."""
    return {
        "exact_default": exact_span_metrics(gold, predictions, mode="default"),
        "exact_strict": exact_span_metrics(gold, predictions, mode="strict"),
        "relaxed": relaxed_span_metrics(gold, predictions),
        "token_level": token_level_metrics(gold, predictions),
        "error_taxonomy": classify_errors(gold, predictions),
    }
