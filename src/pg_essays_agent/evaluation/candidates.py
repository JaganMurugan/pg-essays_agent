
import json
import logging
import os
from pathlib import Path
from typing import Protocol

from pg_essays_agent.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    """Interface required by the candidate collector."""

    def run(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        ...


def collect_candidates(
    retriever: Retriever,
    input_path: Path,
    output_path: Path,
    top_k: int = 20,
) -> int:
    """
    Retrieve candidate chunks for every benchmark query.

    Returns:
        Number of benchmark queries processed.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Benchmark file not found: {input_path}"
        )

    benchmark = []
    seen_query_ids = set()

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number}"
                )

            query_id = record.get("query_id")
            query = record.get("query")

            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(
                    f"Missing or invalid query_id at line {line_number}"
                )

            if query_id in seen_query_ids:
                raise ValueError(
                    f"Duplicate query_id: {query_id}"
                )

            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    f"Missing or invalid query at line {line_number}"
                )

            seen_query_ids.add(query_id)
            benchmark.append(record)

    if not benchmark:
        raise ValueError("Benchmark file contains no queries")

    logger.info(
        "Collecting candidates for %d benchmark queries",
        len(benchmark),
    )

    output_records = []

    for index, record in enumerate(benchmark, start=1):
        query_id = record["query_id"]
        query = record["query"]

        logger.info(
            "Processing query %d/%d: %s",
            index,
            len(benchmark),
            query_id,
        )

        results = retriever.run(
            query=query,
            top_k=top_k,
        )

        candidates = []

        for rank, result in enumerate(results, start=1):
            metadata = result.metadata

            candidates.append(
                {
                    "rank": rank,
                    "id": result.id,
                    "score": result.score,
                    "essay_id": metadata.get("essay_id"),
                    "title": metadata.get("title"),
                    "chunk_index": metadata.get("chunk_index"),
                    "text": metadata.get("text"),
                }
            )

        output_record = {
            **record,
            "candidates": candidates,
        }

        output_records.append(output_record)

    # Write to a temporary file first, then atomically replace
    # the destination to avoid leaving a partially written file.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for record in output_records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        os.replace(temporary_path, output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    logger.info(
        "Candidate collection completed. Output: %s",
        output_path,
    )

    return len(output_records)
