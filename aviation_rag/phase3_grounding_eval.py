from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_jsonl
from .schemas import FinalOutput


def evaluate_grounding_gold(
    phase3_rows: list[FinalOutput | dict[str, Any]],
    gold_path: Path,
) -> dict[str, Any]:
    gold_rows = read_jsonl(gold_path)
    if not gold_rows:
        return {"gold_rows": 0, "pass_rate": None}

    lookup: dict[str, dict[str, Any]] = {}
    lookup_by_query: dict[str, dict[str, Any]] = {}
    for row in phase3_rows:
        if hasattr(row, "model_dump"):
            payload = row.model_dump()
        elif isinstance(row, dict):
            payload = row
        else:
            continue
        lookup[str(payload.get("query_id", ""))] = payload
        lookup_by_query[str(payload.get("query_raw", ""))] = payload

    eval_rows = []
    for gold in gold_rows:
        query_id = str(gold.get("query_id", ""))
        query_raw = str(gold.get("query_raw", ""))
        result = lookup.get(query_id) or lookup_by_query.get(query_raw, {})
        answer = str(result.get("answer", ""))
        citations = result.get("citations", []) or []
        risk = float(result.get("hallucination_risk", 1.0) or 1.0)
        checks = {
            "non_empty_answer": len(answer.strip()) > 0 if gold.get("require_non_empty_answer", True) else True,
            "min_citations": len(citations) >= int(gold.get("min_citations", 1)),
            "max_risk": risk <= float(gold.get("max_hallucination_risk", 1.0)),
        }
        eval_rows.append(
            {
                "query_id": query_id,
                "query_raw": gold.get("query_raw"),
                "citation_count": len(citations),
                "hallucination_risk": round(risk, 4),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    passed = sum(row["passed"] for row in eval_rows)
    return {
        "gold_path": str(gold_path),
        "gold_rows": len(eval_rows),
        "pass_rate": round(passed / len(eval_rows), 4),
        "gold_eval_rows": eval_rows,
    }
