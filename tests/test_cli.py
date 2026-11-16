
from unittest.mock import Mock, patch

from pg_essays_agent.cli import build_parser, main


from unittest.mock import Mock, patch

from pg_essays_agent.cli import build_parser, main
from pg_essays_agent.retrieval.models import RetrievalResult

from unittest.mock import Mock, patch

from pg_essays_agent.cli import build_parser, main
from pg_essays_agent.generation.models import (
    GenerationResult,
    Source,
)



def test_ingest_command_parsing():
    parser = build_parser()

    args = parser.parse_args(["ingest"])

    assert args.command == "ingest"
    assert args.fresh is False


def test_fresh_argument_parsing():
    parser = build_parser()

    args = parser.parse_args(["ingest", "--fresh"])

    assert args.command == "ingest"
    assert args.fresh is True


@patch("pg_essays_agent.cli.create_ingestion_pipeline")
def test_main_runs_ingestion(mock_create_pipeline, monkeypatch, capsys):
    pipeline = Mock()
    pipeline.run.return_value = Mock(
        total_chunks=100,
        previously_completed=20,
        newly_processed=80,
    )
    mock_create_pipeline.return_value = pipeline

    monkeypatch.setattr("sys.argv", ["pg-essays-agent", "ingest"])

    main()

    pipeline.run.assert_called_once_with(fresh=False)

    output = capsys.readouterr().out
    assert "Total chunks: 100" in output
    assert "Newly processed: 80" in output


@patch("pg_essays_agent.cli.create_ingestion_pipeline")
def test_main_fresh_ingestion(mock_create_pipeline, monkeypatch):
    pipeline = Mock()
    pipeline.run.return_value = Mock(
        total_chunks=100,
        previously_completed=0,
        newly_processed=100,
    )
    mock_create_pipeline.return_value = pipeline

    monkeypatch.setattr(
        "sys.argv",
        ["pg-essays-agent", "ingest", "--fresh"],
    )

    main()

    pipeline.run.assert_called_once_with(fresh=True)


def test_query_command_parsing():
    parser = build_parser()

    args = parser.parse_args([
        "query",
        "startup product-market fit",
        "--top-k",
        "3",
    ])

    assert args.command == "query"
    assert args.query == "startup product-market fit"
    assert args.top_k == 3


@patch("pg_essays_agent.cli.create_retrieval_pipeline")
def test_main_runs_retrieval(mock_create_pipeline, monkeypatch, capsys):
    pipeline = Mock()

    pipeline.run.return_value = [
        RetrievalResult(
            id="essay-1-0",
            score=0.95,
            metadata={
                "title": "How to Start a Startup",
                "text": "Solve a real problem.",
                "essay_id": "essay-1",
                "chunk_index": 0,
            },
        )
    ]

    mock_create_pipeline.return_value = pipeline

    monkeypatch.setattr(
        "sys.argv",
        [
            "pg-essays-agent",
            "query",
            "startup advice",
            "--top-k",
            "3",
        ],
    )

    main()

    pipeline.run.assert_called_once_with(
        query="startup advice",
        top_k=3,
    )

    output = capsys.readouterr().out

    assert "How to Start a Startup" in output
    assert "Solve a real problem." in output
    assert "0.9500" in output


@patch("pg_essays_agent.cli.create_retrieval_pipeline")
def test_query_displays_message_when_no_results(
    mock_create_pipeline,
    monkeypatch,
    capsys,
):
    pipeline = Mock()
    pipeline.run.return_value = []
    mock_create_pipeline.return_value = pipeline

    monkeypatch.setattr(
        "sys.argv",
        ["pg-essays-agent", "query", "unknown topic"],
    )

    main()

    assert "No matching essay chunks found." in capsys.readouterr().out


def test_query_rejects_invalid_top_k(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["pg-essays-agent", "query", "startup", "--top-k", "0"],
    )

    with patch("pg_essays_agent.cli.create_retrieval_pipeline") as factory:
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 2

        factory.assert_not_called()





def test_ask_command_parsing():
    parser = build_parser()

    args = parser.parse_args([
        "ask",
        "What does doing things that don't scale mean?",
        "--top-k",
        "5",
    ])

    assert args.command == "ask"
    assert args.top_k == 5
    assert "scale" in args.query


@patch("pg_essays_agent.cli.create_rag_pipeline")
def test_main_runs_ask(mock_create_pipeline, monkeypatch, capsys):
    source = Source(
        chunk_id="essay-1-0",
        title="Do Things that Don't Scale",
        essay_id="essay-1",
        chunk_index=0,
        text="Startups should personally acquire users.",
        retrieval_score=0.95,
    )

    expected_result = GenerationResult(
        answer="Startups should personally acquire their first users. [S1]",
        sources=[source],
        model="gemini-3.8-flash",
    )

    pipeline = Mock()
    pipeline.run.return_value = expected_result
    mock_create_pipeline.return_value = pipeline

    monkeypatch.setattr(
        "sys.argv",
        [
            "pg-essays-agent",
            "ask",
            "What does doing things that don't scale mean?",
            "--top-k",
            "5",
        ],
    )

    main()

    pipeline.run.assert_called_once_with(
        query="What does doing things that don't scale mean?",
        top_k=5,
    )

    output = capsys.readouterr().out

    assert "Answer:" in output
    assert "personally acquire their first users" in output
    assert "Do Things that Don't Scale" in output
    assert "[S1]" in output
    assert "gemini-3.8-flash" in output


@patch("pg_essays_agent.cli.create_rag_pipeline")
def test_ask_displays_answer_without_sources(
    mock_create_pipeline,
    monkeypatch,
    capsys,
):
    pipeline = Mock()
    pipeline.run.return_value = GenerationResult(
        answer="No relevant passages found.",
        sources=[],
        model="gemini-3.8-flash",
    )
    mock_create_pipeline.return_value = pipeline

    monkeypatch.setattr(
        "sys.argv",
        ["pg-essays-agent", "ask", "Unknown question"],
    )

    main()

    output = capsys.readouterr().out

    assert "No relevant passages found." in output
    assert "Sources:" not in output


def test_ask_rejects_invalid_top_k(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["pg-essays-agent", "ask", "Startup advice", "--top-k", "0"],
    )

    with patch("pg_essays_agent.cli.create_rag_pipeline") as factory:
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 2

        factory.assert_not_called()
