
import pytest

from pg_essays_agent.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["A", "C", "E"]

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_precision_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["A", "C", "E"]

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_recall_returns_zero_when_no_relevant_ids():
    assert recall_at_k(["A", "B"], [], 2) == 0.0


def test_precision_returns_zero_when_no_relevant_ids():
    assert precision_at_k(["A", "B"], [], 2) == 0.0


def test_recall_returns_zero_when_no_relevant_results():
    assert recall_at_k(["A", "B"], ["C", "D"], 2) == 0.0


def test_precision_returns_zero_when_no_relevant_results():
    assert precision_at_k(["A", "B"], ["C", "D"], 2) == 0.0


def test_mrr():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["C", "D"]

    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_mrr_returns_zero_when_no_relevant_results():
    assert mean_reciprocal_rank(["A", "B"], ["C"]) == 0.0


def test_mrr_returns_zero_when_no_relevant_ids():
    assert mean_reciprocal_rank(["A", "B"], []) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_recall_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        recall_at_k(["A"], ["A"], k)


@pytest.mark.parametrize("k", [0, -1])
def test_precision_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        precision_at_k(["A"], ["A"], k)


def test_duplicate_retrieved_ids_are_not_double_counted():
    assert recall_at_k(["A", "A", "B"], ["A", "B"], 3) == 1.0
    assert precision_at_k(["A", "A", "B"], ["A", "B"], 3) == pytest.approx(2 / 3)


def test_precision_uses_k_as_denominator_when_fewer_results_returned():
    assert precision_at_k(["A"], ["A", "B"], 5) == pytest.approx(1 / 5)
