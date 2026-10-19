"""Command-line interface for running Cascade-XID on CSV flow data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.pipeline.cascade import load_cascade_from_artifacts, read_csv_rows

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the configured Tier-1/gate/Tier-2 cascade and write JSONL records."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="score a CSV with the full cascade")
    run_parser.add_argument("--input", required=True, type=Path)
    run_parser.add_argument("--gate-profile", default="p10")
    run_parser.add_argument("--out", required=True, type=Path)
    run_parser.add_argument("--dataset", default=None)
    run_parser.add_argument("--config", default=None)
    run_parser.add_argument("--batch-size", type=int, default=None)
    run_parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    dataset = args.dataset or config.data.dataset
    if not dataset:
        logger.error("data.dataset is null; pass --dataset or configure data.dataset")
        return 2
    if not args.input.is_file():
        logger.error("input CSV does not exist: %s", args.input)
        return 2
    batch_size = args.batch_size or config.pipeline.batch_size
    try:
        pipeline = load_cascade_from_artifacts(
            dataset,
            args.gate_profile,
            config=config,
            tier2_concurrency=args.concurrency,
        )
        count = pipeline.write_jsonl(
            read_csv_rows(args.input, batch_size),
            args.out,
            batch_size=batch_size,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as error:
        logger.error("Cascade run stopped: %s", error)
        return 1
    logger.info("Wrote %d schema-validated records to %s", count, args.out)
    logger.info("Run counters saved to %s", pipeline.run_stats_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
