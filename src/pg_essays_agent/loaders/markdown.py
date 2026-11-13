
import re
from collections.abc import Iterator
from pathlib import Path

from pg_essays_agent.exceptions import DataLoadingError
from pg_essays_agent.models import Essay


MONTHS = (
    "January|February|March|April|May|June|"
    "July|August|September|October|November|December"
)

MONTH_YEAR = re.compile(rf"^(?:{MONTHS})\s+\d{{4}}$")

FRONTMATTER = re.compile(
    r"\A---[ \t]*\n.*?\n---[ \t]*\n",
    re.DOTALL,
)


class MarkdownLoader:
    """Loads essays from a directory containing Markdown files."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load(self) -> Iterator[Essay]:
        root = self.directory

        if not root.is_dir():
            raise DataLoadingError(
                f"Markdown source directory does not exist: {root}"
            )

        files = [
            path
            for path in sorted(root.rglob("*.md"))
            if path.name.lower() != "readme.md"
            and not any(
                part.startswith(".")
                for part in path.relative_to(root).parts
            )
        ]

        if not files:
            raise DataLoadingError(
                f"No Markdown files found under {root}"
            )

        for path in files:
            try:
                raw = FRONTMATTER.sub(
                    "",
                    path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ),
                    count=1,
                )

                lines = raw.lstrip().splitlines()

                title = (
                    path.stem
                    .replace("-", " ")
                    .replace("_", " ")
                    .strip()
                )

                if lines and lines[0].startswith("# "):
                    title = lines[0][2:].strip() or title
                    lines = lines[1:]

                date = next(
                    (
                        line.strip()
                        for line in lines[:6]
                        if MONTH_YEAR.match(line.strip())
                    ),
                    "",
                )

                yield Essay(
                    title=title,
                    date=date,
                    text="\n".join(lines),
                )

            except OSError as exc:
                raise DataLoadingError(
                    f"Could not read Markdown file {path}: {exc}"
                ) from exc
