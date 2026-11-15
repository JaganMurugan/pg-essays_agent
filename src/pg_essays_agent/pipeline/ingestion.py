
import logging
import time
from dataclasses import dataclass

from pg_essays_agent.checkpoint.manager import CheckpointManager
from pg_essays_agent.config import Settings
from pg_essays_agent.embeddings.base import EmbeddingProvider
from pg_essays_agent.loaders.base import EssayLoader
from pg_essays_agent.vectorstores.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    total_chunks: int
    previously_completed: int
    newly_processed: int


class IngestionPipeline:
    """Coordinates the end-to-end essay ingestion process."""

    def __init__(
        self,
        loader: EssayLoader,
        chunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        checkpoint: CheckpointManager,
        settings: Settings,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.checkpoint = checkpoint
        self.settings = settings

    def run(self, fresh: bool = False) -> IngestionResult:
        """Load, chunk, embed and upsert essays with resumable progress."""

        # 1. Load essays
        logger.info("Loading essays...")
        essays = list(self.loader.load())

        logger.info("Loaded %d essays", len(essays))

        if not essays:
            logger.warning("No essays found")
            return IngestionResult(0, 0, 0)

        # 2. Create chunks
        logger.info("Creating chunks...")
        chunks = self.chunker.build_chunks(essays)

        logger.info("Created %d chunks", len(chunks))

        if not chunks:
            logger.warning("No chunks created")
            return IngestionResult(0, 0, 0)

        # 3. Calculate fingerprint
        fingerprint = self.checkpoint.fingerprint(chunks)

        # 4. Load or clear checkpoint
        if fresh:
            self.checkpoint.clear()

        done = self.checkpoint.load(fingerprint)

        # Only count completed IDs that exist in the current input.
        chunk_ids = {chunk.id for chunk in chunks}
        completed_ids = done & chunk_ids

        todo = [
            chunk for chunk in chunks
            if chunk.id not in completed_ids
        ]

        logger.info(
            "%d chunks total | %d already done | %d remaining",
            len(chunks),
            len(completed_ids),
            len(todo),
        )

        if not todo:
            logger.info("Nothing to ingest")
            return IngestionResult(
                total_chunks=len(chunks),
                previously_completed=len(completed_ids),
                newly_processed=0,
            )

        # 5. Process remaining chunks in batches
        started = time.monotonic()
        processed = 0

        for start in range(0, len(todo), self.settings.batch_size):
            batch = todo[start:start + self.settings.batch_size]

            texts = [chunk.embed_text for chunk in batch]

            # 6. Generate embeddings
            vectors = self.embedder.embed(texts)

            # Validate provider output before sending it to Pinecone.
            if len(vectors) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} embeddings, "
                    f"received {len(vectors)}"
                )

            for vector in vectors:
                if len(vector) != self.settings.embedding_dimension:
                    raise ValueError(
                        "Embedding dimension mismatch: "
                        f"expected {self.settings.embedding_dimension}, "
                        f"received {len(vector)}"
                    )

            # 7. Build Pinecone payload
            payload = [
                {
                    "id": chunk.id,
                    "values": vector,
                    "metadata": chunk.metadata,
                }
                for chunk, vector in zip(batch, vectors, strict=True)
            ]

            # 8. Upsert into vector database
            # The Pinecone adapter handles transient retries.
            self.vector_store.upsert(payload)

            # 9. Save progress only after successful upsert.
            done.update(chunk.id for chunk in batch)
            self.checkpoint.save(fingerprint, done)

            processed += len(batch)

            elapsed = max(time.monotonic() - started, 1e-9)
            rate = processed / elapsed
            remaining = len(todo) - processed
            eta = int(remaining / rate) if rate > 0 else 0

            logger.info(
                "Progress %d/%d | %.1f chunks/s | ETA %ds",
                len(done),
                len(chunks),
                rate,
                eta,
            )

        logger.info("Ingestion completed successfully")

        return IngestionResult(
            total_chunks=len(chunks),
            previously_completed=len(completed_ids),
            newly_processed=processed,
        )
