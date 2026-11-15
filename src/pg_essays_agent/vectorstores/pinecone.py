
import logging
import time

from pinecone import Pinecone, ServerlessSpec

from pg_essays_agent.config import Settings
from pg_essays_agent.exceptions import VectorStoreError

from pg_essays_agent.utils.retry import with_retries

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings

        try:
            self.client = Pinecone(api_key=settings.pinecone_api_key)
            self._ensure_index()
            self.index = self.client.Index(settings.pinecone_index)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize Pinecone: {exc}"
            ) from exc

    def _ensure_index(self) -> None:
        settings = self.settings
        index_name = settings.pinecone_index

        existing_indexes = self.client.list_indexes().names()

        if index_name not in existing_indexes:
            logger.info("Creating Pinecone index: %s", index_name)

            self.client.create_index(
                name=index_name,
                dimension=settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )

        deadline = time.monotonic() + 300

        while True:
            description = self.client.describe_index(index_name)

            if description.status["ready"]:
                break

            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Pinecone index {index_name} was not ready "
                    "after five minutes"
                )

            time.sleep(2)

        if description.dimension != settings.embedding_dimension:
            raise VectorStoreError(
                f"Index {index_name} has dimension "
                f"{description.dimension}, but the embedding model "
                f"produces {settings.embedding_dimension}. "
                "Use a different index or correct its configuration."
            )

    def upsert(self, vectors: list[dict]) -> None:
        if not vectors:
            return

        try:
            with_retries(
            lambda: self.index.upsert(
                vectors=vectors,
                namespace=self.settings.pinecone_namespace,
            ),
            what="Pinecone upsert",
        )
            
        except Exception as exc:
            raise VectorStoreError(
                f"Pinecone upsert failed: {exc}"
            ) from exc

    
    def query(self, vector: list[float], top_k: int = 5) -> list[dict]:
        try:
            response = with_retries(
                lambda: self.index.query(
                    vector=vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=self.settings.pinecone_namespace,
                ),
                what="Pinecone query",
            )

            return [
                {
                    "id": match.id,
                    "score": float(match.score),
                    "metadata": match.metadata or {},
                }
                for match in response.matches
            ]

        except Exception as exc:
            raise VectorStoreError(
                f"Pinecone query failed: {exc}"
            ) from exc


    def count(self) -> int:
        try:
            stats = with_retries(
                lambda: self.index.describe_index_stats(),
                what="Pinecone index statistics",
            )

            namespace_stats = stats.namespaces.get(
                self.settings.pinecone_namespace
            )

            return namespace_stats.vector_count if namespace_stats else 0

        except Exception as exc:
            raise VectorStoreError(
                f"Could not retrieve Pinecone vector count: {exc}"
            ) from exc
