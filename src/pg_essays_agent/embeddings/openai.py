
from openai import OpenAI

from pg_essays_agent.config import Settings
from pg_essays_agent.exceptions import EmbeddingError


class OpenAIEmbeddingProvider:
    """Generates embeddings using the OpenAI embeddings API."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY is required for the OpenAI provider."
            )

        self.settings = settings

        # Disable SDK retries because we'll manage retry behavior
        # centrally when we extract the retry utility.
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            max_retries=0,
            timeout=60,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=texts,
            )

            vectors = [
                item.embedding
                for item in response.data
            ]

            self._validate(vectors, len(texts))

            return vectors

        except EmbeddingError:
            raise

        except Exception as exc:
            raise EmbeddingError(
                f"OpenAI embedding generation failed: {exc}"
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
