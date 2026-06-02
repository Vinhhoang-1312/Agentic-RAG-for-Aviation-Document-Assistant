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

from aviation_rag.io_utils import read_jsonl
from aviation_rag.schemas import MiddleAgentOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate San phase 2 retrieval artifact against the shared contract."
    )
    parser.add_argument(
        "--phase2-artifact",
        type=str,
        default="artifacts/phase2_san_retrieval_output.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.phase2_artifact)
    rows = read_jsonl(artifact_path)
    if not rows:
        raise ValueError(f"No rows found in {artifact_path}")

    validated = [MiddleAgentOutput.model_validate(row) for row in rows]
    intents = sorted({row.predicted_intent for row in validated})
    print(f"Validated {len(validated)} San phase 2 rows from {artifact_path}")
    print(f"Intents present: {', '.join(intents)}")


if __name__ == "__main__":
    main()
