"""
Reciprocal Rank Fusion (RRF) implementation for hybrid search.

Combines results from multiple retrieval sources into a single
ranked list using the RRF algorithm.
"""

from typing import Any, Callable


def reciprocal_rank_fusion(
    result_sets: list[list[dict[str, Any]]],
    k: int = 60,
    key_fn: Callable[[dict], str] | None = None,
) -> list[dict[str, Any]]:
    """
    Combine multiple ranked result sets using Reciprocal Rank Fusion.
    
    RRF score = Σ 1 / (k + rank_i) for each result set
    
    Args:
        result_sets: List of result lists from different sources
        k: RRF constant (default 60, per original paper)
        key_fn: Function to extract unique key from result item
                (default: uses 'id' or generates from content hash)
    
    Returns:
        Fused and re-ranked results with RRF scores
    """
    if key_fn is None:
        key_fn = _default_key_fn

    # Accumulate RRF scores
    scores: dict[str, dict[str, Any]] = {}

    for result_set in result_sets:
        for rank, item in enumerate(result_set):
            key = key_fn(item)

            if key not in scores:
                scores[key] = {
                    "item": item,
                    "rrf_score": 0.0,
                    "source_ranks": {},
                }

            # RRF formula
            rrf_contribution = 1.0 / (k + rank + 1)
            scores[key]["rrf_score"] += rrf_contribution

            # Track source ranks for debugging
            source = item.get("source", "unknown")
            scores[key]["source_ranks"][source] = rank + 1

    # Sort by RRF score descending
    sorted_results = sorted(
        scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    # Return items with RRF scores added
    return [
        {
            **entry["item"],
            "rrf_score": entry["rrf_score"],
            "source_ranks": entry["source_ranks"],
        }
        for entry in sorted_results
    ]


def _default_key_fn(item: dict[str, Any]) -> str:
    """Generate a unique key for a result item."""
    # Try common ID fields
    if item.get("id"):
        return str(item["id"])
    if item.get("tx_hash"):
        return item["tx_hash"]
    if item.get("hash"):
        return item["hash"]

    # Fall back to content hash
    import hashlib
    import json

    # Remove volatile fields for consistent hashing
    stable_item = {
        k: v for k, v in item.items()
        if k not in ("score", "rrf_score", "source_ranks", "source")
    }

    content = json.dumps(stable_item, sort_keys=True, default=str)
    return hashlib.md5(content.encode()).hexdigest()


def weighted_rrf(
    result_sets: list[list[dict[str, Any]]],
    weights: list[float],
    k: int = 60,
    key_fn: Callable[[dict], str] | None = None,
) -> list[dict[str, Any]]:
    """
    Weighted Reciprocal Rank Fusion.
    
    Allows assigning different importance weights to different sources.
    
    Args:
        result_sets: List of result lists from different sources
        weights: Weight for each result set (should sum to 1.0)
        k: RRF constant
        key_fn: Function to extract unique key from result item
    
    Returns:
        Fused and re-ranked results with weighted RRF scores
    """
    if len(weights) != len(result_sets):
        raise ValueError("Number of weights must match number of result sets")

    if key_fn is None:
        key_fn = _default_key_fn

    scores: dict[str, dict[str, Any]] = {}

    for weight, result_set in zip(weights, result_sets):
        for rank, item in enumerate(result_set):
            key = key_fn(item)

            if key not in scores:
                scores[key] = {
                    "item": item,
                    "rrf_score": 0.0,
                    "source_ranks": {},
                }

            # Weighted RRF
            rrf_contribution = weight / (k + rank + 1)
            scores[key]["rrf_score"] += rrf_contribution

            source = item.get("source", "unknown")
            scores[key]["source_ranks"][source] = rank + 1

    sorted_results = sorted(
        scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    return [
        {
            **entry["item"],
            "rrf_score": entry["rrf_score"],
            "source_ranks": entry["source_ranks"],
        }
        for entry in sorted_results
    ]


def combine_with_reranking(
    result_sets: list[list[dict[str, Any]]],
    reranker_fn: Callable[[str, list[dict]], list[tuple[dict, float]]],
    query: str,
    k: int = 60,
    rerank_top_n: int = 100,
) -> list[dict[str, Any]]:
    """
    Combine results with RRF then apply neural reranking.
    
    Two-stage retrieval:
    1. RRF fusion to get candidates
    2. Neural reranking on top candidates
    
    Args:
        result_sets: List of result lists from different sources
        reranker_fn: Function that takes (query, docs) and returns [(doc, score), ...]
        query: Original search query
        k: RRF constant
        rerank_top_n: Number of candidates to rerank
    
    Returns:
        Reranked results
    """
    # Stage 1: RRF fusion
    fused = reciprocal_rank_fusion(result_sets, k=k)

    # Take top candidates for reranking
    candidates = fused[:rerank_top_n]

    if not candidates:
        return []

    # Stage 2: Neural reranking
    reranked = reranker_fn(query, candidates)

    # Sort by reranker score
    reranked.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            **doc,
            "rerank_score": score,
        }
        for doc, score in reranked
    ]
