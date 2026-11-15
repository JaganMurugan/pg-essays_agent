
from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Interface that every vector database adapter must implement."""

    def upsert(self, vectors: list[dict]) -> None:
        ...

    def query(self, vector: list[float], top_k: int = 5) -> list[dict]:
        ...

    def count(self) -> int:
        ...
