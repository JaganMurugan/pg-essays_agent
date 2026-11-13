
import logging
import re
import unicodedata
from collections.abc import Iterable

import tiktoken

from pg_essays_agent.config import Settings
from pg_essays_agent.models import Chunk, Essay


logger = logging.getLogger(__name__)


class TextChunker:
    """Cleans essays and converts them into token-aware chunks."""

    def __init__(self, settings: Settings) -> None:
        self.chunk_tokens = settings.chunk_tokens
        self.overlap_tokens = settings.overlap_tokens

        # Load the tokenizer once per chunker instance.
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens using the cl100k_base tokenizer."""
        return len(self.encoder.encode(text))

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize Unicode, line endings and whitespace."""

        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        """
        Split an oversized paragraph into sentence groups.

        If an individual sentence is still too long,
        split it into smaller word groups.
        """

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        pieces: list[str] = []
        current: list[str] = []

        for sentence in sentences:

            # Handle a single sentence exceeding the token budget.
            if self.count_tokens(sentence) > self.chunk_tokens:

                if current:
                    pieces.append(" ".join(current))
                    current = []

                words = sentence.split()

                step = max(1, int(self.chunk_tokens * 0.6))

                pieces.extend(
                    " ".join(words[i:i + step])
                    for i in range(0, len(words), step)
                )

                continue

            candidate = " ".join(current + [sentence])

            if current and self.count_tokens(candidate) > self.chunk_tokens:
                pieces.append(" ".join(current))
                current = []

            current.append(sentence)

        if current:
            pieces.append(" ".join(current))

        return pieces

    def chunk_text(self, text: str) -> list[str]:
        """
        Create paragraph-aware chunks with token-based overlap.
        """

        units: list[tuple[str, int]] = []

        # First, split the document into paragraphs.
        paragraphs = re.split(r"\n+", text)

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            token_count = self.count_tokens(paragraph)

            if token_count <= self.chunk_tokens:
                units.append((paragraph, token_count))

            else:
                smaller_pieces = self._split_long_paragraph(paragraph)

                units.extend(
                    (piece, self.count_tokens(piece))
                    for piece in smaller_pieces
                )

        chunks: list[str] = []

        current: list[tuple[str, int]] = []
        current_tokens = 0

        for unit, token_count in units:

            # Start a new chunk if adding this unit exceeds the budget.
            if current and current_tokens + token_count > self.chunk_tokens:

                chunks.append(
                    "\n\n".join(text for text, _ in current)
                )

                # Carry the previous chunk's tail as overlap.
                carry: list[tuple[str, int]] = []
                carry_tokens = 0

                for previous in reversed(current):

                    if carry_tokens + previous[1] > self.overlap_tokens:
                        break

                    carry.insert(0, previous)
                    carry_tokens += previous[1]

                current = carry
                current_tokens = carry_tokens

            current.append((unit, token_count))
            current_tokens += token_count

        if current:
            chunks.append(
                "\n\n".join(text for text, _ in current)
            )

        return chunks

    @staticmethod
    def slugify(title: str, fallback: str) -> str:
        """Convert an essay title into a stable identifier."""

        return (
            re.sub(r"[^a-z0-9]+", "-", title.lower())
            .strip("-")[:80]
            or fallback
        )

    def build_chunks(self, essays: Iterable[Essay]) -> list[Chunk]:
        """Convert essays into Chunk objects with deterministic IDs."""

        chunks: list[Chunk] = []
        seen: set[str] = set()
        processed_essays = 0

        for index, essay in enumerate(essays):

            title = essay.title.strip() or f"essay-{index}"
            text = self.clean_text(essay.text)

            if not text:
                logger.warning(
                    "Skipping essay with no text: %r",
                    title,
                )
                continue

            slug = self.slugify(title, f"essay-{index}")

            # Prevent duplicate titles from generating duplicate IDs.
            if slug in seen:
                slug = f"{slug}-{index}"

            seen.add(slug)

            pieces = self.chunk_text(text)

            processed_essays += 1

            for chunk_index, piece in enumerate(pieces):

                metadata: dict[str, str | int] = {
                    "essay_id": slug,
                    "title": title,
                    "chunk_index": chunk_index,
                    "n_chunks": len(pieces),
                    "text": piece,
                }

                if essay.date:
                    metadata["date"] = essay.date

                chunks.append(
                    Chunk(
                        id=f"{slug}-{chunk_index}",
                        embed_text=f"{title}\n\n{piece}",
                        metadata=metadata,
                    )
                )

        if not chunks:
            raise ValueError(
                "No chunks were produced. Is the data source empty?"
            )

        logger.info(
            "Built %d chunks from %d essays",
            len(chunks),
            processed_essays,
        )

        return chunks
