from collections.abc import Iterator
from pg_essays_agent.config import Settings
from pg_essays_agent.exceptions import DataLoadingError
from pg_essays_agent.models import Essay

class HuggingFaceLoader:
    """Loads Paul Graham essays from a Hugging Face dataset."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self) -> Iterator[Essay]:
        try:
            from datasets import load_dataset

            dataset = load_dataset(self.settings.hf_dataset)

            split = (
                dataset["train"]
                if "train" in dataset
                else next(iter(dataset.values()))
                 
            )

            missing = {"title", "text"} - set(split.column_names)

            if missing:
                raise DataLoadingError(
                    f"{self.settings.hf_dataset} is missing "
                    f"expected columns: {sorted(missing)}"
                )

            for row in split:
                yield Essay(
                    title=(row.get("title") or "").strip(),
                    date=(row.get("date") or "").strip(),
                    text=row.get("text") or "",
                )

        except DataLoadingError:
            raise

        except Exception as exc:
            raise DataLoadingError(
                f"Failed to load Hugging Face dataset "
                f"{self.settings.hf_dataset!r}: {exc}"
            ) from exc
