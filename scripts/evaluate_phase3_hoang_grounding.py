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

import pandas as pd

from aviation_rag.io_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Hoang phase 3 grounded answer quality from phase3 artifact."
    )
    parser.add_argument(
        "--phase3-artifact",
        type=str,
        default="artifacts/phase3_hoang_grounded_answer_output.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.phase3_artifact))
    if not rows:
        raise ValueError(f"No rows found in {args.phase3_artifact}")

    evaluation_rows = []
    for row in rows:
        citations = row.get("citations", [])
        grounding_report = row.get("grounding_report") or {}
        evaluation_rows.append(
            {
                "query_id": row.get("query_id"),
                "has_citation": len(citations) > 0,
                "hallucination_risk": row.get("hallucination_risk", 1.0),
                "overlap_ratio": grounding_report.get("overlap_ratio", 0.0),
            }
        )

    frame = pd.DataFrame(evaluation_rows)
    print("\nHoang Phase 3 Grounding Metrics:")
    print(f"Citation coverage: {frame['has_citation'].mean():.2%}")
    print(f"Avg hallucination risk: {frame['hallucination_risk'].mean():.4f}")
    print(f"Avg overlap ratio: {frame['overlap_ratio'].mean():.4f}")
    print(f"Rows evaluated: {len(frame)}")


if __name__ == "__main__":
    main()
