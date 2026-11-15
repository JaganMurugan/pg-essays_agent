
import argparse
import logging

from pg_essays_agent.application import create_ingestion_pipeline


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pg-essays-agent",
        description="Paul Graham Essays RAG ingestion application",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest essays into the vector database",
    )

    ingest_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear the ingestion checkpoint and ingest everything again",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        pipeline = create_ingestion_pipeline()
        result = pipeline.run(fresh=args.fresh)

        print("\nIngestion completed.")
        print(f"Total chunks: {result.total_chunks}")
        print(f"Previously completed: {result.previously_completed}")
        print(f"Newly processed: {result.newly_processed}")


if __name__ == "__main__":
    main()
