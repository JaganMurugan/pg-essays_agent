
from unittest.mock import patch

from pg_essays_agent.config import Settings
from pg_essays_agent.application import create_rag_pipeline
from pg_essays_agent.pipeline.rag import RAGPipeline


@patch("pg_essays_agent.application.build_generation_provider")
@patch("pg_essays_agent.application.create_retrieval_pipeline")
def test_create_rag_pipeline(
    mock_retrieval,
    mock_generation,
):
    settings = Settings(gemini_api_key="test-key")

    retriever = object()
    generator = object()

    mock_retrieval.return_value = retriever
    mock_generation.return_value = generator

    pipeline = create_rag_pipeline(settings)

    assert isinstance(pipeline, RAGPipeline)
    assert pipeline.retriever is retriever
    assert pipeline.generator is generator

    mock_retrieval.assert_called_once_with(settings)
    mock_generation.assert_called_once_with(settings)


@patch("pg_essays_agent.application.load_settings")
@patch("pg_essays_agent.application.build_generation_provider")
@patch("pg_essays_agent.application.create_retrieval_pipeline")
def test_create_rag_pipeline_loads_settings(
    mock_retrieval,
    mock_generation,
    mock_load_settings,
):
    settings = Settings(gemini_api_key="test-key")

    mock_load_settings.return_value = settings

    pipeline = create_rag_pipeline()

    assert isinstance(pipeline, RAGPipeline)
    mock_load_settings.assert_called_once()
    mock_retrieval.assert_called_once_with(settings)
    mock_generation.assert_called_once_with(settings)
