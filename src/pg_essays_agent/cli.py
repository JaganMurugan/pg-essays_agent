
import argparse
import logging

from pg_essays_agent.application import (
    create_ingestion_pipeline,
    create_retrieval_pipeline,
)

from pathlib import Path

from pg_essays_agent.evaluation.candidates import collect_candidates

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pg-essays-agent",
        description="Paul Graham Essays RAG application",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # Ingestion command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest essays into the vector database",
    )

    ingest_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear the checkpoint and ingest everything again",
    )

    # Retrieval command
    query_parser = subparsers.add_parser(
        "query",
        help="Search for relevant essay chunks",
    )

    query_parser.add_argument(
        "query",
        help="Natural-language search query",
    )

    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)",
    )

    

    # Candidate collection command
    candidate_parser = subparsers.add_parser(
        "collect-candidates",
        help="Retrieve candidate chunks for evaluation queries",
    )

    candidate_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/evaluation/benchmark.jsonl"),
        help="Benchmark JSONL file",
    )

    candidate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/candidates.jsonl"),
        help="Candidate output JSONL file",
    )

    candidate_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of candidate chunks per query",
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

    elif args.command == "query":
        if args.top_k <= 0:
            parser.error("--top-k must be greater than zero")

        pipeline = create_retrieval_pipeline()
        results = pipeline.run(
            query=args.query,
            top_k=args.top_k,
        )

        if not results:
            print("No matching essay chunks found.")
            return

        print(f"\nTop {len(results)} results:\n")

        for rank, result in enumerate(results, start=1):
            metadata = result.metadata

            title = metadata.get("title", "Untitled")
            text = metadata.get("text", "")
            essay_id = metadata.get("essay_id", "Unknown")
            chunk_index = metadata.get("chunk_index", "Unknown")

            print(f"{rank}. {title}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Essay ID: {essay_id}")
            print(f"   Chunk: {chunk_index}")
            print(f"   {text}")
            print("-" * 80)

    elif args.command == "collect-candidates":
        if args.top_k <= 0:
            parser.error("--top-k must be greater than zero")

        pipeline = create_retrieval_pipeline()

        count = collect_candidates(
            retriever=pipeline,
            input_path=args.input,
            output_path=args.output,
            top_k=args.top_k,
        )

        print("\nCandidate collection completed.")
        print(f"Queries processed: {count}")
        print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
