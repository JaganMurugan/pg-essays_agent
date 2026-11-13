
from pg_essays_agent.config import load_settings
from pg_essays_agent.models import Essay
from pg_essays_agent.chunking.chunker import TextChunker


def test_chunker():

    settings = load_settings()
    chunker = TextChunker(settings)

    essay = Essay(
        title="Test Essay",
        date="January 2025",
        text=(
            "Artificial intelligence is changing technology. "
            "Machine learning is a major part of artificial intelligence. "
            "Deep learning uses neural networks.\n\n"
            "Retrieval augmented generation combines retrieval "
            "with language models. It is useful for knowledge-intensive "
            "applications."
        ),
    )

    # Test token counting.
    token_count = chunker.count_tokens(essay.text)
    print("Token count:", token_count)

    # Test cleaning.
    cleaned = chunker.clean_text(
        "Hello   world.\r\nThis is a test."
    )
    print("Cleaned text:", repr(cleaned))

    # Test chunk creation.
    chunks = chunker.build_chunks([essay])

    for chunk in chunks:
        print("\nChunk ID:", chunk.id)
        print("Embedding text:", chunk.embed_text)
        print("Metadata:", chunk.metadata)

        assert chunk.metadata["title"] == "Test Essay"
        assert chunk.metadata["essay_id"] == "test-essay"
        assert chunk.embed_text.startswith("Test Essay\n\n")

    assert len(chunks) >= 1

    # Test deterministic IDs.
    chunks_again = chunker.build_chunks([essay])

    assert [c.id for c in chunks] == [c.id for c in chunks_again]

    print("\nChunker test passed")


if __name__ == "__main__":
    test_chunker()
