
import logging

from pg_essays_agent.generation.base import GenerationProvider
from pg_essays_agent.generation.models import GenerationResult
from pg_essays_agent.retrieval.pipeline import RetrievalPipeline


logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrate retrieval and answer generation."""

    def __init__(
        self,
        retriever: RetrievalPipeline,
        generator: GenerationProvider,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    def run(
        self,
        query: str,
        top_k: int = 5,
    ) -> GenerationResult:
        """Retrieve relevant context and generate an answer."""

        logger.info("Starting RAG pipeline")

        # 1. Retrieve relevant essay chunks.
        retrieved_chunks = self.retriever.run(
            query=query,
            top_k=top_k,
        )

        logger.info(
            "Retrieved %d chunks for generation",
            len(retrieved_chunks),
        )

        # 2. Generate an answer using retrieved context.
        result = self.generator.generate(
            query=query,
            context=retrieved_chunks,
        )

        logger.info("RAG pipeline completed")

        return result
