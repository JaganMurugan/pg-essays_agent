
from unittest.mock import Mock

import pytest

from pg_essays_agent.config import Settings
from pg_essays_agent.models import Chunk
from pg_essays_agent.pipeline.ingestion import IngestionPipeline


def make_settings(tmp_path):
    return Settings(
        batch_size=2,
        checkpoint_path=tmp_path / "checkpoint.json",
        embedding_dimension=3,
    )


def make_chunks():
    return [
        Chunk(
            id=f"essay-{i}",
            embed_text=f"Essay content {i}",
            metadata={"title": f"Essay {i}"},
        )
        for i in range(3)
    ]


def make_pipeline(tmp_path):
    settings = make_settings(tmp_path)

    loader = Mock()
    loader.load.return_value = ["essay-1", "essay-2"]

    chunker = Mock()
    chunker.build_chunks.return_value = make_chunks()

    embedder = Mock()
    embedder.embed.side_effect = lambda texts: [
    [0.1, 0.2, 0.3] for _ in texts
]

    vector_store = Mock()
    checkpoint = Mock()
    checkpoint.fingerprint.return_value = "test-fingerprint"
    checkpoint.load.return_value = set()

    pipeline = IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        checkpoint=checkpoint,
        settings=settings,
    )

    return pipeline, loader, chunker, embedder, vector_store, checkpoint


def test_ingestion_processes_all_chunks(tmp_path):
    pipeline, loader, chunker, embedder, store, checkpoint = (
        make_pipeline(tmp_path)
    )

    result = pipeline.run()

    assert result.total_chunks == 3
    assert result.previously_completed == 0
    assert result.newly_processed == 3

    loader.load.assert_called_once()
    chunker.build_chunks.assert_called_once()

    assert embedder.embed.call_count == 2
    assert store.upsert.call_count == 2
    assert checkpoint.save.call_count == 2


def test_ingestion_resumes_from_checkpoint(tmp_path):
    pipeline, _, _, embedder, store, checkpoint = make_pipeline(tmp_path)

    checkpoint.load.return_value = {"essay-0", "essay-1"}

    result = pipeline.run()

    assert result.previously_completed == 2
    assert result.newly_processed == 1
    assert embedder.embed.call_count == 1
    assert store.upsert.call_count == 1


def test_fresh_ingestion_clears_checkpoint(tmp_path):
    pipeline, _, _, _, _, checkpoint = make_pipeline(tmp_path)

    pipeline.run(fresh=True)

    checkpoint.clear.assert_called_once()


def test_checkpoint_is_saved_only_after_successful_upsert(tmp_path):
    pipeline, _, _, _, store, checkpoint = make_pipeline(tmp_path)

    store.upsert.side_effect = RuntimeError("Pinecone unavailable")

    with pytest.raises(RuntimeError, match="Pinecone unavailable"):
        pipeline.run()

    checkpoint.save.assert_not_called()


def test_embedding_count_mismatch_raises_error(tmp_path):
    pipeline, _, _, embedder, store, checkpoint = make_pipeline(tmp_path)

    embedder.embed.side_effect = None
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with pytest.raises(ValueError, match="Expected 2 embeddings"):
        pipeline.run()

    store.upsert.assert_not_called()
    checkpoint.save.assert_not_called()


def test_embedding_dimension_mismatch_raises_error(tmp_path):
    pipeline, _, _, embedder, store, checkpoint = make_pipeline(tmp_path)

    embedder.embed.side_effect = None
    embedder.embed.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    with pytest.raises(ValueError, match="dimension mismatch"):
        pipeline.run()

    with pytest.raises(ValueError, match="dimension mismatch"):
        pipeline.run()

    store.upsert.assert_not_called()
    checkpoint.save.assert_not_called()
