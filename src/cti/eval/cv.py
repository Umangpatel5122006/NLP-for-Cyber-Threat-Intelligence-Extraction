"""Report-grouped cross-validation for CTI sequence tagging.

Ensures that entire intelligence reports (doc_id) remain clustered together in either
train or validation folds, preventing cross-sentence document leakage.
The official test split is strictly excluded and asserted against.
"""

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[3]
TEST_IDS_PATH = ROOT / "data" / "splits" / "test_report_ids.json"


def get_locked_test_ids() -> Set[str]:
    """Retrieve test split report IDs to guarantee they are never included in CV."""
    if TEST_IDS_PATH.exists():
        with open(TEST_IDS_PATH, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def get_report_grouped_splits(
    doc_ids: List[str],
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    repeats: int = 1,
) -> List[Tuple[List[int], List[int]]]:
    """Generate train/val index splits grouped strictly by doc_id.
    
    Parameters:
        doc_ids: list of report identifiers, one per sentence in the dataset.
        n_splits: number of folds (default 5).
        shuffle: whether to shuffle report order before partitioning.
        seed: random seed for reproducibility.
        repeats: number of repeated K-fold iterations with different seeds.
        
    Returns:
        List of (train_indices, val_indices) pairs.
    """
    test_ids = get_locked_test_ids()
    unique_reports = sorted(list(set(doc_ids)))

    # Strict audit: test split reports must never be present
    contaminated = set(unique_reports) & test_ids
    if contaminated:
        raise PermissionError(
            f"Test split reports detected in CV dataset: {contaminated}. "
            "Cross-validation must only operate over train + dev reports."
        )

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(unique_reports) < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds with only {len(unique_reports)} distinct reports."
        )

    # Map each report to its list of sentence indices in the dataset
    report_to_indices: Dict[str, List[int]] = {}
    for idx, doc_id in enumerate(doc_ids):
        report_to_indices.setdefault(doc_id, []).append(idx)

    all_folds = []

    for r in range(repeats):
        current_seed = seed + (r * 1000)
        rng = random.Random(current_seed)
        shuffled_reports = list(unique_reports)
        if shuffle:
            rng.shuffle(shuffled_reports)

        # Distribute reports across K bins greedily by sentence count for balance
        bins: List[List[str]] = [[] for _ in range(n_splits)]
        bin_sizes = [0] * n_splits

        for rep in shuffled_reports:
            # Place in the bin with the smallest current sentence count
            min_bin = min(range(n_splits), key=lambda b: bin_sizes[b])
            bins[min_bin].append(rep)
            bin_sizes[min_bin] += len(report_to_indices[rep])

        for k in range(n_splits):
            val_reports = set(bins[k])
            train_reports = set(shuffled_reports) - val_reports

            train_idx = [
                idx for rep in train_reports for idx in report_to_indices[rep]
            ]
            val_idx = [
                idx for rep in val_reports for idx in report_to_indices[rep]
            ]

            # Invariant check: zero report overlap
            assert not (train_reports & val_reports), "Report leakage between folds!"

            all_folds.append((sorted(train_idx), sorted(val_idx)))

    return all_folds
