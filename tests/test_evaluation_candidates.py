
import json

import pytest

from pg_essays_agent.evaluation.candidates import collect_candidates
from pg_essays_agent.retrieval.models import RetrievalResult


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def run(self, query, top_k=5):
        self.calls.append((query, top_k))

        return [
            RetrievalResult(
                id="chunk-001",
                score=0.95,
                metadata={
                    "essay_id": "essay-001",
                    "title": "Test Essay",
                    "chunk_index": 0,
                    "text": "Example essay content.",
                },
            ),
            RetrievalResult(
                id="chunk-002",
                score=0.85,
                metadata={
                    "essay_id": "essay-001",
                    "title": "Test Essay",
                    "chunk_index": 1,
                    "text": "Another example.",
                },
            ),
        ]


def test_collect_candidates(tmp_path):
    input_path = tmp_path / "benchmark.jsonl"
    output_path = tmp_path / "candidates.jsonl"

    benchmark_record = {
        "query_id": "Q001",
        "query": "What is a startup?",
        "category": "startup_ideas",
        "relevant_chunk_ids": [],
        "label_status": "pending",
    }

    input_path.write_text(
        json.dumps(benchmark_record) + "\n",
        encoding="utf-8",
    )

    retriever = FakeRetriever()

    count = collect_candidates(
        retriever=retriever,
        input_path=input_path,
        output_path=output_path,
        top_k=20,
    )

    assert count == 1
    assert retriever.calls == [("What is a startup?", 20)]
    assert output_path.exists()

    records = [
        json.loads(line)
        for line in output_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 1

    record = records[0]

    assert record["query_id"] == "Q001"
    assert record["label_status"] == "pending"
    assert len(record["candidates"]) == 2

    first_candidate = record["candidates"][0]

    assert first_candidate["rank"] == 1
    assert first_candidate["id"] == "chunk-001"
    assert first_candidate["score"] == 0.95
    assert first_candidate["title"] == "Test Essay"
    assert first_candidate["chunk_index"] == 0


def test_collect_candidates_rejects_invalid_top_k(tmp_path):
    input_path = tmp_path / "benchmark.jsonl"
    input_path.write_text(
        '{"query_id":"Q001","query":"Test"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        collect_candidates(
            retriever=FakeRetriever(),
            input_path=input_path,
            output_path=tmp_path / "output.jsonl",
            top_k=0,
        )


def test_collect_candidates_rejects_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_candidates(
            retriever=FakeRetriever(),
            input_path=tmp_path / "missing.jsonl",
            output_path=tmp_path / "output.jsonl",
        )


def test_collect_candidates_rejects_duplicate_query_ids(tmp_path):
    input_path = tmp_path / "benchmark.jsonl"

    records = [
        {"query_id": "Q001", "query": "Question one"},
        {"query_id": "Q001", "query": "Question two"},
    ]

    input_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate query_id"):
        collect_candidates(
            retriever=FakeRetriever(),
            input_path=input_path,
            output_path=tmp_path / "output.jsonl",
        )


def test_collect_candidates_rejects_invalid_json(tmp_path):
    input_path = tmp_path / "benchmark.jsonl"
    input_path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        collect_candidates(
            retriever=FakeRetriever(),
            input_path=input_path,
            output_path=tmp_path / "output.jsonl",
        )
