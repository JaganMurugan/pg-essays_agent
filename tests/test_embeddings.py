
from unittest.mock import Mock

import pytest

from pg_essays_agent.config import Settings
from pg_essays_agent.embeddings.base import EmbeddingProvider
from pg_essays_agent.embeddings.local import LocalEmbeddingProvider
from pg_essays_agent.exceptions import EmbeddingError


def test_local_provider_implements_interface(monkeypatch):
    settings = Settings(
        embedding_provider="local",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=3,
    )

    fake_model = Mock()
    fake_model.encode.return_value.tolist.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    monkeypatch.setattr(
        "pg_essays_agent.embeddings.local.SentenceTransformer",
        lambda model_name: fake_model,
    )

    provider = LocalEmbeddingProvider(settings)

    assert isinstance(provider, EmbeddingProvider)

    vectors = provider.embed(["First text", "Second text"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3

    fake_model.encode.assert_called_once_with(
        ["First text", "Second text"],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def test_local_provider_rejects_wrong_dimension(monkeypatch):
    settings = Settings(
        embedding_provider="local",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=3,
    )

    fake_model = Mock()
    fake_model.encode.return_value.tolist.return_value = [
        [0.1, 0.2]
    ]

    monkeypatch.setattr(
        "pg_essays_agent.embeddings.local.SentenceTransformer",
        lambda model_name: fake_model,
    )

    provider = LocalEmbeddingProvider(settings)

    with pytest.raises(EmbeddingError, match="dimension"):
        provider.embed(["Test text"])


def test_empty_input_returns_empty_list(monkeypatch):
    settings = Settings(embedding_dimension=3)

    fake_model = Mock()

    monkeypatch.setattr(
        "pg_essays_agent.embeddings.local.SentenceTransformer",
        lambda model_name: fake_model,
    )

    provider = LocalEmbeddingProvider(settings)

    assert provider.embed([]) == []

    fake_model.encode.assert_not_called()



from pg_essays_agent.embeddings.openai import OpenAIEmbeddingProvider


def test_openai_provider_generates_embeddings(monkeypatch):
    settings = Settings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimension=3,
        openai_api_key="test-api-key",
    )

    fake_client = Mock()

    fake_client.embeddings.create.return_value.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6]),
    ]

    monkeypatch.setattr(
        "pg_essays_agent.embeddings.openai.OpenAI",
        lambda **kwargs: fake_client,
    )

    provider = OpenAIEmbeddingProvider(settings)

    vectors = provider.embed(["First text", "Second text"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 3

    fake_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=["First text", "Second text"],
    )


def test_openai_provider_requires_api_key():
    settings = Settings(
        embedding_provider="openai",
        openai_api_key=None,
    )

    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(settings)
