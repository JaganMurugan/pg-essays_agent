
from pg_essays_agent.config import Settings
from pg_essays_agent.embeddings.base import EmbeddingProvider
from pg_essays_agent.embeddings.local import LocalEmbeddingProvider
from pg_essays_agent.embeddings.openai import OpenAIEmbeddingProvider
from pg_essays_agent.exceptions import ConfigurationError


def create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider:
    """Create the configured embedding provider."""

    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(settings)

    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings)

    raise ConfigurationError(
        f"Unsupported embedding provider: "
        f"{settings.embedding_provider}"
    )
