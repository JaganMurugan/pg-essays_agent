from pg_essays_agent.config import Settings
from pg_essays_agent.vectorstores.pinecone import PineconeVectorStore


def create_vector_store(settings: Settings) -> PineconeVectorStore:
    """Create the configured vector store."""

    return PineconeVectorStore(settings)