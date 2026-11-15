
import json

from pg_essays_agent.checkpoint.manager import CheckpointManager
from pg_essays_agent.config import Settings
from pg_essays_agent.models import Chunk


def make_settings(tmp_path):
    return Settings(
        checkpoint_path=tmp_path / "checkpoint.json",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        pinecone_namespace="essays",
        chunk_tokens=400,
        overlap_tokens=60,
    )


def make_chunks():
    return [
        Chunk(
            id="essay-one-0",
            embed_text="Title: Essay One\n\nFirst chunk",
            metadata={"title": "Essay One"},
        ),
        Chunk(
            id="essay-one-1",
            embed_text="Title: Essay One\n\nSecond chunk",
            metadata={"title": "Essay One"},
        ),
    ]


def test_fingerprint_is_deterministic(tmp_path):
    manager = CheckpointManager(make_settings(tmp_path))

    first = manager.fingerprint(make_chunks())
    second = manager.fingerprint(make_chunks())

    assert first == second


def test_fingerprint_changes_when_chunk_content_changes(tmp_path):
    manager = CheckpointManager(make_settings(tmp_path))

    original = manager.fingerprint(make_chunks())

    changed_chunks = [
        Chunk(
            id="essay-one-0",
            embed_text="Modified content",
            metadata={"title": "Essay One"},
        )
    ]

    changed = manager.fingerprint(changed_chunks)

    assert original != changed


def test_fingerprint_changes_when_settings_change(tmp_path):
    settings_a = make_settings(tmp_path)
    settings_b = Settings(
        checkpoint_path=tmp_path / "checkpoint.json",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
        pinecone_namespace="essays",
        chunk_tokens=400,
        overlap_tokens=60,
    )

    fingerprint_a = CheckpointManager(settings_a).fingerprint(make_chunks())
    fingerprint_b = CheckpointManager(settings_b).fingerprint(make_chunks())

    assert fingerprint_a != fingerprint_b


def test_save_and_load_checkpoint(tmp_path):
    manager = CheckpointManager(make_settings(tmp_path))
    fingerprint = manager.fingerprint(make_chunks())

    completed = {"essay-one-0", "essay-one-1"}

    manager.save(fingerprint, completed)
    loaded = manager.load(fingerprint)

    assert loaded == completed


def test_mismatched_fingerprint_starts_fresh(tmp_path):
    manager = CheckpointManager(make_settings(tmp_path))

    manager.save("old-fingerprint", {"essay-one-0"})

    loaded = manager.load("new-fingerprint")

    assert loaded == set()


def test_missing_checkpoint_returns_empty_set(tmp_path):
    manager = CheckpointManager(make_settings(tmp_path))

    assert manager.load("some-fingerprint") == set()


def test_corrupted_checkpoint_returns_empty_set(tmp_path):
    manager = CheckpointManager(make_settings(tmp_path))
    manager.path.write_text("{invalid json", encoding="utf-8")

    assert manager.load("some-fingerprint") == set()


def test_checkpoint_uses_original_json_format(tmp_path):
    manager = CheckpointManager(make_settings(tmp_path))

    manager.save("test-fingerprint", {"essay-one-1", "essay-one-0"})

    data = json.loads(manager.path.read_text(encoding="utf-8"))

    assert data == {
        "fingerprint": "test-fingerprint",
        "done": ["essay-one-0", "essay-one-1"],
    }


def test_clear_removes_checkpoint(tmp_path):
    manager = CheckpointManager(make_settings(tmp_path))
    manager.save("test-fingerprint", {"essay-one-0"})

    manager.clear()

    assert not manager.path.exists()
