from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from aviation_rag.config import Settings, ensure_artifact_dirs
from aviation_rag.io_utils import read_jsonl
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Hoang phase 1 intent-aware routing artifact from raw user queries."
    )
    parser.add_argument("--query", type=str, default=None, help="Single raw query.")
    parser.add_argument(
        "--queries-jsonl",
        type=str,
        default=None,
        help="JSONL file with `query_raw` and optional `query_id`.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output artifact path.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["bm25", "semantic", "hybrid", "metadata_first"],
        help="Optional manual strategy override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.query and not args.queries_jsonl:
        raise ValueError("Provide --query or --queries-jsonl.")

    settings = Settings()
    ensure_artifact_dirs(settings)
    phase1 = Phase1HoangIntentRouting(settings)
    output_path = Path(args.output) if args.output else settings.phase1_output_path

    rows_written = 0
    if args.query:
        output = phase1.build_output(args.query, top_k=args.top_k, strategy=args.strategy)
        phase1.write_output(output, output_path)
        rows_written += 1

    if args.queries_jsonl:
        for row in read_jsonl(Path(args.queries_jsonl)):
            query_raw = row.get("query_raw", "")
            if not query_raw:
                continue
            output = phase1.build_output(
                query_raw=query_raw,
                query_id=row.get("query_id"),
                top_k=args.top_k,
                strategy=args.strategy,
            )
            phase1.write_output(output, output_path)
            rows_written += 1

    print(f"Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
