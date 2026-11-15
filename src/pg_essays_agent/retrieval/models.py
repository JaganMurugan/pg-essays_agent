
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A retrieved essay chunk and its similarity score."""

    id: str
    score: float
    metadata: dict[str, Any]
