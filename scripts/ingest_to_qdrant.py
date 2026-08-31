#!/usr/bin/env python3
"""
CLI entry-point for the Qdrant ingestion pipeline.

All heavy-lifting lives in ``services.ingest_service.pipeline``.

Usage:
    PYTHONPATH=src python scripts/ingest_to_qdrant.py --source kb --strategy parent_child
    PYTHONPATH=src python scripts/ingest_to_qdrant.py --source kb --recreate
    PYTHONPATH=src python scripts/ingest_to_qdrant.py --source markdown --strategy fixed
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from services.ingest_service.pipeline import STRATEGY_MAP, LOADER_MAP, run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest document chunks into Qdrant Cloud",
    )
    parser.add_argument(
        "--source",
        choices=list(LOADER_MAP.keys()),
        default="kb",
        help="Document source (default: kb = data/knowledge_base/)",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGY_MAP.keys()),
        default="parent_child",
        help="Chunking strategy (default: parent_child)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting",
    )
    args = parser.parse_args()

    run_ingest(
        source=args.source,
        strategy=args.strategy,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
