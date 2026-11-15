
from unittest.mock import Mock, patch

from pg_essays_agent.cli import build_parser, main


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
