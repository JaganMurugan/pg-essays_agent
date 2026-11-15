
from unittest.mock import Mock

import pytest

from pg_essays_agent.config import Settings
from pg_essays_agent.exceptions import VectorStoreError
from pg_essays_agent.vectorstores.pinecone import PineconeVectorStore
from pg_essays_agent.vectorstores import pinecone as pinecone_module


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

    fake_match.id = "test-chunk-001"

    results = store.query([0.1, 0.2, 0.3], top_k=5)

    assert results == [
        {
            "id": "test-chunk-001",
            "score": 0.95,
            "metadata": {
                "title": "Test essay",
                "text": "Example",
            },
        }
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

def test_existing_index_is_reused(monkeypatch):
    settings = make_settings()

    fake_client = Mock()
    fake_client.list_indexes.return_value.names.return_value = [
        "test-index"
    ]

    description = Mock()
    description.status = {"ready": True}
    description.dimension = 3

    fake_client.describe_index.return_value = description

    monkeypatch.setattr(
        pinecone_module,
        "Pinecone",
        lambda **kwargs: fake_client,
    )

    store = PineconeVectorStore(settings)

    fake_client.create_index.assert_not_called()
    fake_client.Index.assert_called_once_with("test-index")


def test_missing_index_is_created(monkeypatch):
    settings = make_settings()

    fake_client = Mock()
    fake_client.list_indexes.return_value.names.return_value = []

    description = Mock()
    description.status = {"ready": True}
    description.dimension = 3

    fake_client.describe_index.return_value = description

    monkeypatch.setattr(
        pinecone_module,
        "Pinecone",
        lambda **kwargs: fake_client,
    )

    PineconeVectorStore(settings)

    fake_client.create_index.assert_called_once()


def test_index_dimension_mismatch_raises_error(monkeypatch):
    settings = make_settings()

    fake_client = Mock()
    fake_client.list_indexes.return_value.names.return_value = [
        "test-index"
    ]

    description = Mock()
    description.status = {"ready": True}
    description.dimension = 1024

    fake_client.describe_index.return_value = description

    monkeypatch.setattr(
        pinecone_module,
        "Pinecone",
        lambda **kwargs: fake_client,
    )

    with pytest.raises(VectorStoreError, match="dimension"):
        PineconeVectorStore(settings)

def test_upsert_retries_transient_failure(monkeypatch):
    from pg_essays_agent.vectorstores import pinecone as pinecone_module

    fake_index = Mock()
    fake_index.upsert.side_effect = [
        ConnectionError("temporary failure"),
        None,
    ]

    store = PineconeVectorStore.__new__(PineconeVectorStore)
    store.settings = make_settings()
    store.index = fake_index

    monkeypatch.setattr(
        "pg_essays_agent.utils.retry.time.sleep",
        lambda _: None,
    )

    store.upsert([
        {
            "id": "essay-0",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"title": "Test"},
        }
    ])

    assert fake_index.upsert.call_count == 2

    def test_query_preserves_id_score_and_metadata(vector_store):
        vector_store.index.query.return_value = Mock(
            matches=[
                Mock(
                    id="essay-1-0",
                    score=0.95,
                    metadata={
                        "title": "Startup Ideas",
                        "text": "Build something people want.",
                    },
                )
            ]
        )

        results = vector_store.query(
            vector=[0.1, 0.2, 0.3],
            top_k=5,
        )

        assert results == [
            {
                "id": "essay-1-0",
                "score": 0.95,
                "metadata": {
                    "title": "Startup Ideas",
                    "text": "Build something people want.",
                },
            }
        ]