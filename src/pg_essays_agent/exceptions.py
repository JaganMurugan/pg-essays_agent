class PGEssaysAgentError(Exception):
    """Base exception for the PG Essays Agent application"""

class ConfigurationError(PGEssaysAgentError):
    """Raised when application configuration is invalid"""

class DataLoadingError(PGEssaysAgentError):
    """Raised when source data cannot be loaded"""


class ChunkingError(PGEssaysAgentError):
    """Raised when text chunking fails"""


class EmbeddingError(PGEssaysAgentError):
    """ Raised when generating embedding fails"""

class VectorStoreError(PGEssaysAgentError):
    """Raised when interacting with the vector database fails"""

class CheckpointError(PGEssaysAgentError):
    """Raised when checkpoint persistence or recovery fails."""

class GenerationError(PGEssaysAgentError):
    """Raised when LLM answer generation fails."""
