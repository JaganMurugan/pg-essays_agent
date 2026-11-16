
import logging

from pg_essays_agent.checkpoint.manager import CheckpointManager
from pg_essays_agent.chunking.chunker import TextChunker
from pg_essays_agent.config import Settings, load_settings
from pg_essays_agent.embeddings.factory import create_embedding_provider
from pg_essays_agent.loaders.huggingface import HuggingFaceLoader
from pg_essays_agent.pipeline.ingestion import IngestionPipeline
from pg_essays_agent.vectorstores.factory import create_vector_store

from pg_essays_agent.retrieval.pipeline import RetrievalPipeline

from pg_essays_agent.generation.factory import (
    create_generation_provider as build_generation_provider,
)

from pg_essays_agent.generation.factory import (
    create_generation_provider as build_generation_provider,
)
from pg_essays_agent.pipeline.rag import RAGPipeline

def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_ingestion_pipeline(
    settings: Settings | None = None,
) -> IngestionPipeline:
    """Construct the ingestion pipeline and all its dependencies."""

    if settings is None:
        settings = load_settings()

    configure_logging(settings)

    loader = HuggingFaceLoader(settings)
    chunker = TextChunker(settings)
    embedder = create_embedding_provider(settings)
    vector_store = create_vector_store(settings)
    checkpoint = CheckpointManager(settings)

    return IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        checkpoint=checkpoint,
        settings=settings,
    )

def create_retrieval_pipeline(
    settings: Settings | None = None,
) -> RetrievalPipeline:
    """Construct the retrieval pipeline and its dependencies."""

    if settings is None:
        settings = load_settings()

    configure_logging(settings)

    embedder = create_embedding_provider(settings)
    vector_store = create_vector_store(settings)

    return RetrievalPipeline(
        embedder=embedder,
        vector_store=vector_store,
    )

def create_generation_provider(
    settings: Settings | None = None,
):
    """Construct the configured answer generation provider."""

    if settings is None:
        settings = load_settings()

    configure_logging(settings)

    return build_generation_provider(settings)

def create_rag_pipeline(
    settings: Settings | None = None,
) -> RAGPipeline:
    """Construct the complete RAG application pipeline."""

    if settings is None:
        settings = load_settings()

    configure_logging(settings)

    retriever = create_retrieval_pipeline(settings)
    generator = build_generation_provider(settings)

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
    )
