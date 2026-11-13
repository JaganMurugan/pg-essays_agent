
from pg_essays_agent.config import Settings
from pg_essays_agent.models import Essay
from pg_essays_agent.chunking.chunker import TextChunker


def test_long_paragraph():
    settings = Settings(chunk_tokens=50, overlap_tokens=10)
    chunker = TextChunker(settings)

    text = " ".join(
        ["Artificial intelligence is transforming technology."] * 30
    )

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1

    print("Long paragraph test passed:", len(chunks), "chunks")


def test_overlap():
    settings = Settings(chunk_tokens=30, overlap_tokens=10)
    chunker = TextChunker(settings)

    text = "\n\n".join(
        f"Paragraph number {i} contains some meaningful information."
        for i in range(1, 8)
    )

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1

    # Check whether consecutive chunks share content.
    assert any(
        set(chunks[i].split()) & set(chunks[i + 1].split())
        for i in range(len(chunks) - 1)
    )

    print("Overlap test passed")


def test_duplicate_titles():
    settings = Settings()
    chunker = TextChunker(settings)

    essays = [
        Essay("Test Essay", "", "First essay content."),
        Essay("Test Essay", "", "Second essay content."),
    ]

    chunks = chunker.build_chunks(essays)

    ids = [chunk.id for chunk in chunks]

    assert len(ids) == len(set(ids))

    print("Duplicate title test passed:", ids)


if __name__ == "__main__":
    test_long_paragraph()
    test_overlap()
    test_duplicate_titles()

    print("All edge-case tests passed")
