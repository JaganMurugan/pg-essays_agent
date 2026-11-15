
import hashlib
import json
import logging
import os
from pathlib import Path

from pg_essays_agent.config import Settings
from pg_essays_agent.models import Chunk

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages ingestion progress and checkpoint compatibility."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.checkpoint_path

    def fingerprint(self, chunks: list[Chunk]) -> str:
        """
        Generate a SHA-256 fingerprint of the settings and chunks.

        Changing the embedding model, dimension, namespace, chunking
        configuration, chunk IDs, or embedding text invalidates the
        previous checkpoint.
        """
        settings = self.settings

        config = (
            f"{settings.embedding_model}|"
            f"{settings.embedding_dimension}|"
            f"{settings.pinecone_namespace}|"
            f"{settings.chunk_tokens}|"
            f"{settings.overlap_tokens}"
        )

        hasher = hashlib.sha256(config.encode())

        for chunk in chunks:
            hasher.update(chunk.id.encode())
            hasher.update(b"\0")
            hasher.update(chunk.embed_text.encode())
            hasher.update(b"\0")

        return hasher.hexdigest()

    def load(self, fingerprint: str) -> set[str]:
        """Load completed chunk IDs if the checkpoint is compatible."""

        if not self.path.exists():
            return set()

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))

        except (json.JSONDecodeError, OSError):
            logger.warning("Checkpoint unreadable; starting fresh")
            return set()

        if not isinstance(data, dict):
            logger.warning("Checkpoint has an invalid format; starting fresh")
            return set()

        if data.get("fingerprint") != fingerprint:
            logger.warning(
                "Checkpoint doesn't match current data/settings/model; "
                "starting fresh"
            )
            return set()

        done = data.get("done", [])

        if not isinstance(done, list) or not all(
            isinstance(chunk_id, str) for chunk_id in done
        ):
            logger.warning("Checkpoint progress is invalid; starting fresh")
            return set()

        return set(done)

    def save(self, fingerprint: str, done: set[str]) -> None:
        """
        Persist progress atomically.

        Write to a temporary file first, then replace the original.
        This reduces the risk of a partially written checkpoint.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = self.path.with_suffix(".tmp")

        payload = {
            "fingerprint": fingerprint,
            "done": sorted(done),
        }

        temporary_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        os.replace(temporary_path, self.path)

    def clear(self) -> None:
        """Remove the checkpoint if it exists."""
        self.path.unlink(missing_ok=True)
