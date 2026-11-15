
from pg_essays_agent.config import Settings
from pg_essays_agent.exceptions import ConfigurationError
from pg_essays_agent.vectorstores.pinecone import PineconeVectorStore


def create_vector_store(settings: Settings):
    if settings.vector_store_provider == "pinecone":
        return PineconeVectorStore(settings)

    raise ConfigurationError(
        f"Unsupported vector store provider: "
        f"{settings.vector_store_provider}"
    )
