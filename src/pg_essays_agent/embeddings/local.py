
import logging

from sentence_transformers import SentenceTransformer

from pg_essays_agent.config import Settings
from pg_essays_agent.exceptions import EmbeddingError


logger = logging.getLogger(__name__)


class LocalEmbeddingProvider:
    """Generates embeddings using a local Sentence Transformers model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        logger.info(
            "Loading local embedding model: %s",
            settings.embedding_model,
        )

        self.model = SentenceTransformer(settings.embedding_model)

        # Original implementation used 1024 as the maximum sequence length.
        self.model.max_seq_length = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            vectors = self.model.encode(
                texts,
                batch_size=16,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            result = vectors.tolist()

            self._validate(result, len(texts))

            return result

        except EmbeddingError:
            raise

        except Exception as exc:
            raise EmbeddingError(
                f"Local embedding generation failed: {exc}"
            ) from exc

    def _validate(
        self,
        vectors: list[list[float]],
        expected_count: int,
    ) -> None:
        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"Expected {expected_count} vectors, "
                f"received {len(vectors)}."
            )

        for vector in vectors:
            if len(vector) != self.settings.embedding_dimension:
                raise EmbeddingError(
                    "Unexpected embedding dimension. "
                    f"Expected {self.settings.embedding_dimension}, "
                    f"received {len(vector)}."
                )
