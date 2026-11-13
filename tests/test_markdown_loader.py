
from pathlib import Path
from tempfile import TemporaryDirectory

from pg_essays_agent.loaders.markdown import MarkdownLoader


def test_markdown_loader():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "test-essay.md"

        path.write_text(
            "# Test Essay\n"
            "January 2025\n\n"
            "This is the essay content.",
            encoding="utf-8",
        )

        essays = list(MarkdownLoader(Path(directory)).load())

        print(essays)

        assert len(essays) == 1
        assert essays[0].title == "Test Essay"
        assert essays[0].date == "January 2025"

        print("Markdown loader test passed")


if __name__ == "__main__":
    test_markdown_loader()
