"""Feature caching utilities for fast hyperparameter search and cross-validation."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from cti.models.crf import _sentence_features


def get_cached_sentence_features(
    sentences: List[List[str]],
    feature_config: Optional[Dict[str, Any]] = None,
    cache_path: Optional[Path] = None,
    verbose: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Retrieve pre-extracted features from disk cache, or compute and cache them."""
    if cache_path is not None and Path(cache_path).exists():
        if verbose:
            print(f"Loading cached features from: {cache_path}")
        return joblib.load(cache_path)

    config = feature_config or {}
    if verbose:
        print(f"Computing features for {len(sentences)} sentences...")
    features = [_sentence_features(sentence, **config) for sentence in sentences]

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(features, cache_path, compress=3)
        if verbose:
            print(f"Cached {len(features)} sentence features to: {cache_path}")

    return features
