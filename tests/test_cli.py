
from unittest.mock import Mock, patch

from pg_essays_agent.cli import build_parser, main


from unittest.mock import Mock, patch

from pg_essays_agent.cli import build_parser, main
from pg_essays_agent.retrieval.models import RetrievalResult


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