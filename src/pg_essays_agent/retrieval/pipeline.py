
import logging

from pg_essays_agent.embeddings.base import EmbeddingProvider
from pg_essays_agent.retrieval.models import RetrievalResult
from pg_essays_agent.vectorstores.base import VectorStore

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Retrieve relevant essay chunks for a natural-language query."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def run(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Embed a query and return the most relevant chunks."""

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        logger.info("Embedding retrieval query")

        # 1. Convert the query into an embedding.
        vectors = self.embedder.embed([query])

        if len(vectors) != 1:
            raise ValueError(
                f"Expected one query embedding, received {len(vectors)}"
            )

        query_vector = vectors[0]

        # 2. Search the vector database.
        matches = self.vector_store.query(
            vector=query_vector,
            top_k=top_k,
        )

        # 3. Convert raw matches into application models.
        results = [
            RetrievalResult(
                id=match["id"],
                score=float(match["score"]),
                metadata=match["metadata"],
            )
            for match in matches
        ]

        logger.info("Retrieved %d chunks", len(results))

        return results
