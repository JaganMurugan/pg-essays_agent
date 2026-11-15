
from unittest.mock import Mock

import pytest

from pg_essays_agent.config import Settings
from pg_essays_agent.exceptions import VectorStoreError
from pg_essays_agent.vectorstores.pinecone import PineconeVectorStore


def make_settings():
    return Settings(
        pinecone_api_key="test-key",
        pinecone_index="test-index",
        embedding_dimension=3,
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        pinecone_namespace="test-namespace",
        batch_size=64,
    )


def test_upsert_sends_vectors_to_configured_namespace(monkeypatch):
    fake_index = Mock()
    store = PineconeVectorStore.__new__(PineconeVectorStore)
    store.settings = make_settings()
    store.index = fake_index

    vectors = [
        {"id": "essay-0", "values": [0.1, 0.2, 0.3], "metadata": {"title": "Test"}}
    ]

    store.upsert(vectors)

    fake_index.upsert.assert_called_once_with(
        vectors=vectors,
        namespace="test-namespace",
    )


def test_query_returns_score_and_metadata():
    fake_match = Mock()
    fake_match.score = 0.95
    fake_match.metadata = {"title": "Test essay", "text": "Example"}

    fake_index = Mock()
    fake_index.query.return_value.matches = [fake_match]

    store = PineconeVectorStore.__new__(PineconeVectorStore)
    store.settings = make_settings()
    store.index = fake_index

    results = store.query([0.1, 0.2, 0.3], top_k=5)

    assert results == [
        {"score": 0.95, "title": "Test essay", "text": "Example"}
    ]

    fake_index.query.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        top_k=5,
        include_metadata=True,
        namespace="test-namespace",
    )


def test_count_returns_namespace_vector_count():
    fake_index = Mock()
    fake_index.describe_index_stats.return_value.namespaces = {
        "test-namespace": Mock(vector_count=42)
    }

    store = PineconeVectorStore.__new__(PineconeVectorStore)
    store.settings = make_settings()
    store.index = fake_index

    assert store.count() == 42


def test_upsert_wraps_errors():
    fake_index = Mock()
    fake_index.upsert.side_effect = RuntimeError("connection failed")

    store = PineconeVectorStore.__new__(PineconeVectorStore)
    store.settings = make_settings()
    store.index = fake_index

    with pytest.raises(VectorStoreError, match="upsert failed"):
        store.upsert([{"id": "test", "values": [0.1, 0.2, 0.3]}])
