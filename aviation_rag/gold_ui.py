from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_jsonl
from .phase3_grounding_eval import evaluate_grounding_gold
from .phase2_retrieval_eval import evaluate_gold_retrieval_row, extracted_retrieved_ids


def gold_intent_match(result: dict[str, Any], gold_path: Path) -> dict[str, Any] | None:
    query_raw = str(result.get("query_raw", "")).strip()
    for row in read_jsonl(gold_path):
        if str(row.get("query_raw", "")).strip() == query_raw:
            expected = str(row.get("expected_intent", ""))
            predicted = str(result.get("intent", ""))
            return {
                "query_id": row.get("query_id"),
                "expected_intent": expected,
                "predicted_intent": predicted,
                "correct": predicted == expected,
                "expected_strategy": row.get("expected_strategy"),
                "actual_strategy": (result.get("retrieval_plan") or {}).get("strategy"),
            }
    return None


def gold_grounding_match(result: dict[str, Any], gold_path: Path) -> dict[str, Any] | None:
    query_raw = str(result.get("query_raw", "")).strip()
    gold_defs = read_jsonl(gold_path)
    if not any(str(row.get("query_raw", "")).strip() == query_raw for row in gold_defs):
        return None

    gold_row = {
        "query_id": result.get("query_id"),
        "query_raw": query_raw,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []) or [],
        "hallucination_risk": result.get("hallucination_risk"),
    }
    report = evaluate_grounding_gold([gold_row], gold_path)
    for row in report.get("gold_eval_rows") or []:
        if str(row.get("query_raw", "")).strip() == query_raw:
            return row
    return None


def gold_retrieval_match(result: dict[str, Any], gold_path: Path) -> dict[str, Any] | None:
    query_raw = str(result.get("query_raw", "")).strip()
    gold_row = next(
        (row for row in read_jsonl(gold_path) if str(row.get("query_raw", "")).strip() == query_raw),
        None,
    )
    if gold_row is None:
        return None

    retrieved_ids = extracted_retrieved_ids({"topk_docs": result.get("topk_docs", []) or []})
    diagnostics = result.get("retrieval_diagnostics", {}) or {}
    strategy = (result.get("retrieval_plan") or {}).get("strategy") or diagnostics.get("strategy_requested")
    return evaluate_gold_retrieval_row(
        retrieved_ids,
        gold_row,
        latency_ms=float(diagnostics.get("latency_ms", 0.0) or 0.0),
        strategy=strategy,
    )
