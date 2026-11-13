from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Common interface for all embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert a batch of texts into embedding vectors."""
        ...