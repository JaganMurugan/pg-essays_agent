
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Essay:
    """
    Represents a source essay.

    Attributes:
        title: Essay title.
        date: Publication date, if available.
        text: Original essay content.
    """

    title: str
    date: str
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    Represents a chunk generated from an essay.

    Attributes:
        id: Deterministic identifier for the chunk.
        embed_text: Text sent to the embedding model.
        metadata: Metadata stored alongside the vector.
    """

    id: str
    embed_text: str
    metadata: dict[str, str | int]
