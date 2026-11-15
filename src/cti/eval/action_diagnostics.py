"""Strict action-span diagnostics; no fitting, label repair or model selection."""

import math
from collections import Counter, defaultdict

from cti.eval.metrics import extract_spans


STATUSES = ("exact", "overlap_only", "no_overlap")
LENGTHS = ("1", "2-4", "5-9", "10-19", "20+")


def length_bucket(length):
    if type(length) is not int or length < 1:
        raise ValueError("Span length must be a positive integer.")
    for maximum, label in ((1, "1"), (4, "2-4"), (9, "5-9"), (19, "10-19")):
        if length <= maximum:
            return label
    return "20+"


def overlap(a, b):
    return a[0] == b[0] and max(a[1], b[1]) < min(a[2], b[2])


def action_spans(tags):
    if any(tag not in ("O", "B-ACTION", "I-ACTION") for tag in tags):
        raise ValueError("Diagnostics accept action-layer BIO tags only.")
    return sorted(extract_spans(tags, mode="strict"), key=lambda span: (span[1], span[2]))


def illegal_inside_count(tags):
    action_spans(tags)
    return sum(tag == "I-ACTION" and (i == 0 or tags[i-1] not in ("B-ACTION", "I-ACTION"))
               for i, tag in enumerate(tags))


def gold_statuses(gold, predicted):
    """Per-gold any-overlap diagnostic, not a one-to-one recall/F1 metric."""
    return [(g, "exact" if g in predicted else "overlap_only" if any(overlap(g, p) for p in predicted)
             else "no_overlap") for g in gold]


def pair_spans(gold, predicted):
    """Reserve exact matches, then pair remaining ordered overlaps one-to-one.

    Within one BIO layer each list contains disjoint intervals. Pairing the first
    overlapping residual intervals gives a deterministic maximum-cardinality
    residual matching. Exact-first taxonomy need not equal unconstrained relaxed
    matching; it is not reported as relaxed F1.
    """
    exact = sorted(set(gold) & set(predicted), key=lambda s: (s[1], s[2]))
    gs, ps = [g for g in gold if g not in exact], [p for p in predicted if p not in exact]
    pairs, gi, pi = [], 0, 0
    while gi < len(gs) and pi < len(ps):
        g, p = gs[gi], ps[pi]
        if overlap(g, p):
            pairs.append((g, p)); gi += 1; pi += 1
        elif g[2] <= p[1]:
            gi += 1
        else:
            pi += 1
    paired_gold, paired_pred = {g for g, _ in pairs}, {p for _, p in pairs}
    return {"exact": exact, "boundary_pairs": pairs,
            "missed": [g for g in gs if g not in paired_gold],
            "spurious": [p for p in ps if p not in paired_pred]}


def describe_lengths(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": 0.0, "median_nearest_rank": 0, "p90_nearest_rank": 0, "maximum": 0}
    return {"count": len(values), "mean": sum(values) / len(values),
            "median_nearest_rank": values[math.ceil(.5 * len(values)) - 1],
            "p90_nearest_rank": values[math.ceil(.9 * len(values)) - 1], "maximum": values[-1]}


def _measure(tp, gold, predicted):
    return {"tp": tp, "gold_spans": gold, "predicted_spans": predicted,
            "precision": tp / predicted if predicted else 0.0, "recall": tp / gold if gold else 0.0,
            "f1": 2 * tp / (gold + predicted) if gold + predicted else 0.0}


def diagnose(gold, predicted, relevant):
    if not len(gold) == len(predicted) == len(relevant):
        raise ValueError("Sentence counts must align.")
    if any(type(value) is not bool for value in relevant):
        raise ValueError("Sentence relevance must contain booleans.")
    counts, types, statuses, token_counts = Counter(), Counter(), Counter(), Counter()
    lengths, pred_lengths = [], []
    buckets = {label: Counter({status: 0 for status in STATUSES}) for label in LENGTHS}
    illegal = Counter()
    for truth, guess, is_relevant in zip(gold, predicted, relevant):
        if len(truth) != len(guess):
            raise ValueError("Gold/prediction token counts must align.")
        gs, ps = action_spans(truth), action_spans(guess)
        matched = pair_spans(gs, ps)
        for key in ("exact", "boundary_pairs", "missed", "spurious"):
            counts[key] += len(matched[key])
        counts["gold_spans"] += len(gs); counts["predicted_spans"] += len(ps)
        counts["background_predicted_spans"] += len(ps) if not is_relevant else 0
        counts["spurious_in_relevant_no_gold"] += len(matched["spurious"]) if is_relevant and not gs else 0
        counts["gold_with_multiple_predicted_overlaps"] += sum(sum(overlap(g, p) for p in ps) > 1 for g in gs)
        counts["predicted_with_multiple_gold_overlaps"] += sum(sum(overlap(g, p) for g in gs) > 1 for p in ps)
        for g, status in gold_statuses(gs, ps):
            statuses[status] += 1
            buckets[length_bucket(g[2]-g[1])][status] += 1
        for g, p in matched["boundary_pairs"]:
            kind = "prediction_inside_gold" if g[1] <= p[1] and p[2] <= g[2] else (
                "gold_inside_prediction" if p[1] <= g[1] and g[2] <= p[2] else "crossing_boundary")
            types[kind] += 1
            types["same_start"] += int(g[1] == p[1]); types["same_end"] += int(g[2] == p[2])
        lengths.extend(g[2]-g[1] for g in gs); pred_lengths.extend(p[2]-p[1] for p in ps)
        illegal["gold"] += illegal_inside_count(truth); illegal["predicted"] += illegal_inside_count(guess)
        for g, p in zip(truth, guess):
            token_counts["tp" if g != "O" and p != "O" else "fn" if g != "O" else "fp" if p != "O" else "tn"] += 1
    require_conservation(counts)
    token = {key: token_counts[key] for key in ("tp", "fp", "fn", "tn")}
    token["f1"] = 2 * token["tp"] / (2 * token["tp"] + token["fp"] + token["fn"]) if 2 * token["tp"] + token["fp"] + token["fn"] else 0.0
    return {"exact_strict": _measure(counts["exact"], counts["gold_spans"], counts["predicted_spans"]),
            "taxonomy_exact_first": dict(counts), "boundary_pair_types": dict(types),
            "gold_any_overlap_status": {status: statuses[status] for status in STATUSES},
            "gold_status_by_token_length": {key: dict(value) for key, value in buckets.items()},
            "gold_lengths": describe_lengths(lengths), "predicted_lengths": describe_lengths(pred_lengths),
            "token_action_membership": token, "illegal_inside_starts": dict(illegal)}


def require_conservation(counts):
    if (counts["gold_spans"] != counts["exact"] + counts["boundary_pairs"] + counts["missed"]
            or counts["predicted_spans"] != counts["exact"] + counts["boundary_pairs"] + counts["spurious"]):
        raise ValueError("Span pairing failed conservation checks.")


def compare_gold(gold, baseline, candidate):
    if not len(gold) == len(baseline) == len(candidate):
        raise ValueError("Sentence counts must align.")
    transitions = Counter({f"{a}->{b}": 0 for a in STATUSES for b in STATUSES})
    for truth, before, after in zip(gold, baseline, candidate):
        if not len(truth) == len(before) == len(after):
            raise ValueError("Token counts must align.")
        gs = action_spans(truth)
        a, b = gold_statuses(gs, action_spans(before)), gold_statuses(gs, action_spans(after))
        for (_, status_a), (_, status_b) in zip(a, b):
            transitions[f"{status_a}->{status_b}"] += 1
    return dict(transitions)


def duplicate_annotations(tokens, tags, relevant, doc_ids):
    if not len(tokens) == len(tags) == len(relevant) == len(doc_ids):
        raise ValueError("Sentence metadata must align.")
    groups = defaultdict(list)
    for i, words in enumerate(tokens):
        if len(words) != len(tags[i]):
            raise ValueError("Token counts must align.")
        if words:
            groups[tuple(words)].append(i)
    duplicate = [rows for rows in groups.values() if len(rows) > 1]
    return {"definition": "Identical case-sensitive token sequences within official train only.",
            "duplicate_groups": len(duplicate), "rows_in_duplicate_groups": sum(map(len, duplicate)),
            "cross_report_groups": sum(len({doc_ids[i] for i in rows}) > 1 for rows in duplicate),
            "conflicting_action_tag_groups": sum(len({tuple(tags[i]) for i in rows}) > 1 for rows in duplicate),
            "conflicting_relevance_groups": sum(len({relevant[i] for i in rows}) > 1 for rows in duplicate)}
