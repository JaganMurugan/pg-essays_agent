
from unittest.mock import Mock

from pg_essays_agent.generation.models import (
    GenerationResult,
    Source,
)
from pg_essays_agent.pipeline.rag import RAGPipeline
from pg_essays_agent.retrieval.models import RetrievalResult


def test_rag_pipeline_retrieves_and_generates():
    query = "What does doing things that don't scale mean?"

    retrieved_chunks = [
        RetrievalResult(
            id="essay-1-0",
            score=0.95,
            metadata={
                "title": "Do Things that Don't Scale",
                "essay_id": "essay-1",
                "chunk_index": 0,
                "text": "Startups should personally acquire users.",
            },
        )
    ]

    expected_result = GenerationResult(
        answer="Startups should personally acquire their first users. [S1]",
        sources=[
            Source(
                chunk_id="essay-1-0",
                title="Do Things that Don't Scale",
                essay_id="essay-1",
                chunk_index=0,
                text="Startups should personally acquire users.",
                retrieval_score=0.95,
            )
        ],
        model="gemini-3.8-flash",
    )

    retriever = Mock()
    generator = Mock()

    retriever.run.return_value = retrieved_chunks
    generator.generate.return_value = expected_result

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    result = pipeline.run(query=query, top_k=5)

    retriever.run.assert_called_once_with(
        query=query,
        top_k=5,
    )

    generator.generate.assert_called_once_with(
        query=query,
        context=retrieved_chunks,
    )

    assert result == expected_result
    assert len(result.sources) == 1


def test_rag_pipeline_handles_empty_retrieval():
    query = "Some unknown question"

    expected_result = GenerationResult(
        answer="I couldn't find relevant essay passages to answer this question.",
        sources=[],
        model="gemini-3.8-flash",
    )

    retriever = Mock()
    generator = Mock()

    retriever.run.return_value = []
    generator.generate.return_value = expected_result

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    result = pipeline.run(query=query)

    generator.generate.assert_called_once_with(
        query=query,
        context=[],
    )

    assert result == expected_result
    assert result.sources == []
