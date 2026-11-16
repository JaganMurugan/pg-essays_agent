
from typing import Protocol

from pg_essays_agent.generation.models import GenerationResult
from pg_essays_agent.retrieval.models import RetrievalResult


class GenerationProvider(Protocol):
    """Interface for LLM-based answer generation."""

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
    ) -> GenerationResult:
        """Generate an answer grounded in retrieved context."""
        ...
