import os
from dataclasses import dataclass, field
from pathlib import Path

from pg_essays_agent.exceptions import ConfigurationError

DEFAULT_DATASET = "sgoe19/paul_graham_essays"

@dataclass(frozen=True, slots=True)
class Settings:
    """
    Central application configuration.

    Values are loaded from environment variables
    through the load_settings() factory function.
    """

    # Data Source
    hf_dataset: str = DEFAULT_DATASET

    # Embedding configuration
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024

    # Pinecone configuration
    pinecone_index: str = "pg-essays-bge-m3"
    pinecone_namespace: str = "essays"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Chunking configuration
    chunk_tokens: int = 400  
    overlap_tokens: int = 60

    # Ingestion configuration
    batch_size: int = 64
    checkpoint_path: Path = Path("pg_ingest_checkpoint.json")

    # Logging
    log_level: str = "INFO"

    # Credentials: excluded from repr to avoid accidental exposure
    pinecone_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration after initialization"""

        if self.embedding_provider not in {"local", "openai"}:
            raise ConfigurationError(
                "EMBED_PROVIDER must be 'local' or 'openai'."
                f"Received: {self.embedding_provider!r}"
            )

        if self.chunk_tokens <=0:
            raise ConfigurationError(
                "PG_CHUNK_TOKENS cannot be negative"
            )

        if self.overlap_tokens < 0:
            raise ConfigurationError(
                "PG_OVERLAP_TOKENS cannot be negative."
            )

        if self.overlap_tokens >= self.chunk_tokens:
            raise ConfigurationError(
                "PG_OVERLAP_TOKENS must be smaller thank PG_CHUNK_TOKENS."
            )

        if self.batch_size <= 0:
            raise ConfigurationError(
                "PG_BATCH must be positive"
                
            )

        if self.embedding_dimension <= 0:
            raise ConfigurationError(
                "Embeddign dimension must be positive"
            )

        if not self.pinecone_index.strip():
            raise ConfigurationError(
                "Pincone Index name cannot be empty"
            )

        if not self.pinecone_namespace.strip():
            raise ConfigurationError(
                "Pinecone namespace cannot be empty"
            )

def _load_dotenv() -> None:
    """
    Load .env variables if python-dotenv is installed.

    Environment variables already defiend by the operating
    system take precedence over values in the .env file.
    """

    try:
        from dotenv import find_dotenv, load_dotenv

        dotenv_path = find_dotenv(usecwd=True)

        if dotenv_path:
            load_dotenv(dotenv_path, override=False)

    except ImportError:
        pass

def _get_int(name: str, default: int) -> int:
    """Read and parase a positive integer environment variable"""

    raw = os.getenv(name)

    if raw is None or raw == "":
        return default
    try:
        return int(raw)

    except ValueError as exc:
        raise ConfigurationError(
            f"{name} musst be an integer. Received {raw!r}"
        )

def load_settings() -> Settings:
    """
    Load application settings from environment variables.

    This function is the single entry point for creating
    the application's settings object.
    """

    _load_dotenv()

    provider = os.getenv(
        "EMBED_PROVIDER", "local"
    ).strip().lower()

    if provider == "local":
        model = "BAAI/bge-m3"
        dimension = 1024
        default_index = "pg-essays-bge-m3"

    elif provider == "openai":
        model = "text-embedding-3-small"
        dimension = 1536
        default_index = "pg-essays-openai"

    else:
        raise ConfigurationError(
            f"Unsupported embedding provider: {provider!r}"
        )

    index_name = os.getenv(
        "PINECONE_INDEX", default_index
    )

    return Settings(
        hf_dataset=os.getenv(
            "HF_DATASET", DEFAULT_DATASET
        ),

        embedding_provider=provider,
        embedding_model=model,
        embedding_dimension=dimension,

        pinecone_index=index_name,
        pinecone_namespace=os.getenv(
            "PG_NAMESPACE", "essays"
        ),
        pinecone_cloud=os.getenv(
            "PINECONE_CLOUD", "aws"
        ),
        pinecone_region=os.getenv(
            "PINECONE_REGION", "us-east-1"
        ),

        chunk_tokens=_get_int(
            "PG_CHUNK_TOKENS", 400
        ),
        overlap_tokens=_get_int(
            "PG_OVERLAP_TOKENS", 60
        ),

        batch_size=_get_int(
            "PG_BATCH", 64
        ),

        checkpoint_path=Path(
            os.getenv(
                "PG_CHECKPOINT",
                f"pg_ingest_checkpoint_{index_name}.json"
            )
        ),

        log_level=os.getenv(
            "LOG_LEVEL", "INFO"
        ).upper(),

        pinecone_api_key=os.getenv(
            "PINECONE_API_KEY"
        ),
        openai_api_key=os.getenv(
            "OPENAI_API_KEY"
        ),
    )



