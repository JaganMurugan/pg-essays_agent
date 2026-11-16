
from unittest.mock import Mock

import pytest

from pg_essays_agent.config import Settings
from pg_essays_agent.exceptions import GenerationError
from pg_essays_agent.generation.gemini import (
    GeminiGenerationProvider,
)
from pg_essays_agent.retrieval.models import RetrievalResult


@pytest.fixture
def settings():
    return Settings(
        gemini_api_key="test-key",
        gemini_model="gemini-3.8-flash",
    )


@pytest.fixture
def retrieval_context():
    return [
        RetrievalResult(
            id="essay-1-0",
            score=0.95,
            metadata={
                "title": "Do Things that Don't Scale",
                "essay_id": "essay-1",
                "chunk_index": 0,
                "text": (
                    "Startups should do things that don't scale "
                    "to acquire their first users."
                ),
            },
        )
    ]


def test_successful_generation(settings, retrieval_context):
    client = Mock()
    client.models.generate_content.return_value.candidates = []
    client.models.generate_content.return_value.text = (
    "Startups should do things that don't scale to acquire their first users."
    )

    provider = GeminiGenerationProvider(
        settings=settings,
        client=client,
    )

    result = provider.generate(
        query="What does doing things that don't scale mean?",
        context=retrieval_context,
    )

    assert "first users" in result.answer
    assert result.model == settings.gemini_model
    assert len(result.sources) == 1
    assert result.sources[0].title == "Do Things that Don't Scale"

    client.models.generate_content.assert_called_once()


def test_generation_with_empty_context(settings):
    client = Mock()
    client.models.generate_content.return_value.candidates = []
    client.models.generate_content.return_value.text = ""

    provider = GeminiGenerationProvider(
        settings=settings,
        client=client,
    )

    result = provider.generate(
        query="What is product-market fit?",
        context=[],
    )

    assert result.sources == []
    assert "couldn't find relevant" in result.answer.lower()

    client.models.generate_content.assert_not_called()


def test_generation_rejects_empty_query(settings):
    provider = GeminiGenerationProvider(
        settings=settings,
        client=Mock(),
    )

    with pytest.raises(ValueError, match="Query cannot be empty"):
        provider.generate(
            query="   ",
            context=[],
        )


def test_generation_rejects_missing_api_key():
    settings = Settings(gemini_api_key=None)

    with pytest.raises(
        GenerationError,
        match="GEMINI_API_KEY is required",
    ):
        GeminiGenerationProvider(settings=settings)


def test_generation_rejects_empty_response(
    settings,
    retrieval_context,
    monkeypatch,
):
    client = Mock()
    client.models.generate_content.return_value.candidates = []
    client.models.generate_content.return_value.text = ""

    monkeypatch.setattr(
        "pg_essays_agent.generation.gemini.with_retries",
        lambda fn, what: fn(),
    )

    provider = GeminiGenerationProvider(
        settings=settings,
        client=client,
    )

    with pytest.raises(
        GenerationError,
        match="empty response",
    ):
        provider.generate(
            query="What is a startup?",
            context=retrieval_context,
        )


def test_generation_wraps_api_errors(
    settings,
    retrieval_context,
    monkeypatch,
):
    client = Mock()
    client.models.generate_content.side_effect = RuntimeError(
        "Simulated API failure"
    )

    monkeypatch.setattr(
        "pg_essays_agent.generation.gemini.with_retries",
        lambda fn, what: fn(),
    )

    provider = GeminiGenerationProvider(
        settings=settings,
        client=client,
    )

    with pytest.raises(
        GenerationError,
        match="Simulated API failure",
    ):
        provider.generate(
            query="What is a startup?",
            context=retrieval_context,
        )
