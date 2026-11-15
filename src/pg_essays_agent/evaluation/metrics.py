
from collections.abc import Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Calculate Recall@K."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = retrieved_ids[:k]
    relevant_retrieved = set(retrieved) & relevant

    return len(relevant_retrieved) / len(relevant)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """Calculate Precision@K."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = retrieved_ids[:k]
    relevant_retrieved = set(retrieved) & relevant

    return len(relevant_retrieved) / k


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """Calculate reciprocal rank for a single query."""
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0
