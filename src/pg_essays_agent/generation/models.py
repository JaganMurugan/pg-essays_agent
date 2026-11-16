from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Source:
    """A source passage used to generate an answer."""

    chunk_id: str
    title: str
    essay_id: str
    chunk_index: int
    text: str
    retrieval_score: float


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Generated answer and its supporting sources."""

    answer: str
    sources: list[Source]
    model: str