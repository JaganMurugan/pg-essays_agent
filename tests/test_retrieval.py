
from unittest.mock import Mock

import pytest

from pg_essays_agent.retrieval.pipeline import RetrievalPipeline


@pytest.fixture
def retrieval_components():
    embedder = Mock()
    vector_store = Mock()

    embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    vector_store.query.return_value = [
        {
            "id": "essay-1-0",
            "score": 0.95,
            "metadata": {
                "title": "How to Start a Startup",
                "text": "Start by solving a real problem.",
                "essay_id": "essay-1",
                "chunk_index": 0,
            },
        },
        {
            "id": "essay-2-0",
            "score": 0.87,
            "metadata": {
                "title": "Do Things That Don't Scale",
                "text": "Start by doing things manually.",
                "essay_id": "essay-2",
                "chunk_index": 0,
            },
        },
    ]

    pipeline = RetrievalPipeline(
        embedder=embedder,
        vector_store=vector_store,
    )

    return pipeline, embedder, vector_store


def test_retrieval_returns_ranked_results(retrieval_components):
    pipeline, embedder, vector_store = retrieval_components

    results = pipeline.run("How to build a startup?", top_k=2)

    assert len(results) == 2
    assert results[0].id == "essay-1-0"
    assert results[0].score == 0.95
    assert results[1].score == 0.87

    embedder.embed.assert_called_once_with(
        ["How to build a startup?"]
    )

    vector_store.query.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        top_k=2,
    )


def test_retrieval_strips_query_whitespace(retrieval_components):
    pipeline, embedder, _ = retrieval_components

    pipeline.run("  startup advice  ")

    embedder.embed.assert_called_once_with(["startup advice"])


def test_empty_query_raises_error(retrieval_components):
    pipeline, embedder, vector_store = retrieval_components

    with pytest.raises(ValueError, match="Query cannot be empty"):
        pipeline.run("   ")

    embedder.embed.assert_not_called()
    vector_store.query.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises_error(retrieval_components, top_k):
    pipeline, embedder, vector_store = retrieval_components

    with pytest.raises(ValueError, match="top_k"):
        pipeline.run("startup advice", top_k=top_k)

    embedder.embed.assert_not_called()
    vector_store.query.assert_not_called()


def test_embedding_count_mismatch_raises_error(retrieval_components):
    pipeline, embedder, vector_store = retrieval_components

    embedder.embed.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    with pytest.raises(ValueError, match="Expected one query embedding"):
        pipeline.run("startup advice")

    vector_store.query.assert_not_called()


def test_no_matches_returns_empty_list(retrieval_components):
    pipeline, _, vector_store = retrieval_components

    vector_store.query.return_value = []

    results = pipeline.run("unrelated query")

    assert results == []
