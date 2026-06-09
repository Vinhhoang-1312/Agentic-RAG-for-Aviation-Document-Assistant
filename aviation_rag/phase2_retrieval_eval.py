from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .io_utils import read_jsonl
from .phase2_metrics import RetrievalMetricResult, evaluate_ranking, summarize_metric_rows
from .schemas import InputAgentOutput, MiddleAgentOutput


def _payload(row: MiddleAgentOutput | dict[str, Any]) -> dict[str, Any]:
    if hasattr(row, "model_dump"):
        return row.model_dump()
    return dict(row)


def extracted_retrieved_ids(row: MiddleAgentOutput | dict[str, Any]) -> list[str]:
    payload = _payload(row)
    doc_ids: list[str] = []
    for doc in payload.get("topk_docs", []) or []:
        if isinstance(doc, dict):
            doc_ids.append(str(doc.get("doc_id", "")))
        else:
            doc_ids.append(str(getattr(doc, "doc_id", "")))
    return [doc_id for doc_id in doc_ids if doc_id]


def evaluate_gold_retrieval_row(
    retrieved_ids: list[str],
    gold_row: dict[str, Any],
    *,
    latency_ms: float = 0.0,
    strategy: str | None = None,
) -> dict[str, Any]:
    relevant_ids = [str(doc_id) for doc_id in (gold_row.get("relevant_doc_ids") or []) if str(doc_id)]
    k = int(gold_row.get("k", 5))
    metrics = evaluate_ranking(retrieved_ids, relevant_ids, k=k, latency_ms=latency_ms)
    min_recall = gold_row.get("minimum_recall_at_k")
    if min_recall is None:
        passed = metrics.recall_at_k > 0.0 or metrics.mrr > 0.0
    else:
        passed = metrics.recall_at_k >= float(min_recall)

    first_hit_rank = None
    relevant = set(relevant_ids)
    for index, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            first_hit_rank = index
            break

    return {
        "query_id": gold_row.get("query_id"),
        "query_raw": gold_row.get("query_raw"),
        "strategy": strategy or gold_row.get("expected_strategy"),
        "expected_strategy": gold_row.get("expected_strategy"),
        "k": k,
        "relevant_doc_ids": relevant_ids,
        "retrieved_doc_ids": retrieved_ids[:k],
        "precision_at_k": round(metrics.precision_at_k, 4),
        "recall_at_k": round(metrics.recall_at_k, 4),
        "mrr": round(metrics.mrr, 4),
        "latency_ms": round(metrics.latency_ms, 2),
        "first_hit_rank": first_hit_rank,
        "passed": passed,
    }


def evaluate_retrieval_gold(
    retrieval_rows: list[MiddleAgentOutput | dict[str, Any]],
    gold_path: Path,
    *,
    lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gold_rows = read_jsonl(gold_path)
    if not gold_rows:
        return {"gold_rows": 0, "pass_rate": None}

    by_query_id: dict[str, dict[str, Any]] = {}
    by_query_raw: dict[str, dict[str, Any]] = {}
    for row in retrieval_rows:
        payload = _payload(row)
        by_query_id[str(payload.get("query_id", ""))] = payload
        by_query_raw[str(payload.get("query_raw", "")).strip()] = payload
    if lookup:
        by_query_id.update({key: value for key, value in lookup.items() if key})
        for key, value in lookup.items():
            raw = str(value.get("query_raw", "")).strip()
            if raw:
                by_query_raw[raw] = value

    eval_rows: list[dict[str, Any]] = []
    for gold in gold_rows:
        query_id = str(gold.get("query_id", ""))
        query_raw = str(gold.get("query_raw", "")).strip()
        payload = by_query_id.get(query_id) or by_query_raw.get(query_raw, {})
        retrieved_ids = extracted_retrieved_ids(payload)
        diagnostics = payload.get("retrieval_diagnostics", {}) or {}
        strategy = diagnostics.get("strategy_requested") or payload.get("strategy")
        eval_rows.append(
            evaluate_gold_retrieval_row(
                retrieved_ids,
                gold,
                latency_ms=float(diagnostics.get("latency_ms", 0.0) or 0.0),
                strategy=strategy,
            )
        )

    passed = sum(row["passed"] for row in eval_rows)
    metric_rows = [
        RetrievalMetricResult(
            precision_at_k=row["precision_at_k"],
            recall_at_k=row["recall_at_k"],
            mrr=row["mrr"],
            latency_ms=row["latency_ms"],
        )
        for row in eval_rows
    ]
    return {
        "gold_path": str(gold_path),
        "gold_rows": len(eval_rows),
        "pass_rate": round(passed / len(eval_rows), 4),
        "summary": {key: round(value, 4) for key, value in summarize_metric_rows(metric_rows).items()},
        "gold_eval_rows": eval_rows,
    }


def run_retrieval_gold_benchmark(
    settings: Settings,
    *,
    strategies: list[str] | None = None,
    phase1_agent: Any | None = None,
    retrieval_engine: Any | None = None,
) -> dict[str, Any]:
    from .phase1_hoang_intent_routing import Phase1HoangIntentRouting
    from .phase2_san_faiss_retrieval import Phase2SanFaissRetrieval

    gold_path = settings.phase2_gold_labels_path
    gold_rows = read_jsonl(gold_path)
    if not gold_rows:
        return {"gold_rows": 0, "strategies": {}, "gold_eval_rows": []}

    phase1 = phase1_agent or Phase1HoangIntentRouting(settings)
    retrieval = retrieval_engine or Phase2SanFaissRetrieval(settings)
    strategy_list = strategies or ["bm25", "semantic", "hybrid", "metadata_first", "hybrid_rrf"]

    per_strategy: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    for strategy in strategy_list:
        eval_rows: list[dict[str, Any]] = []
        for gold in gold_rows:
            query_raw = str(gold.get("query_raw", ""))
            query_id = str(gold.get("query_id", ""))
            phase1_output: InputAgentOutput = phase1.build_output(
                query_raw=query_raw,
                query_id=query_id,
                top_k=int(gold.get("k", settings.default_top_k)),
                strategy=strategy,
            )
            if retrieval.available:
                middle_output: MiddleAgentOutput = retrieval.retrieve(phase1_output)
            else:
                middle_output = MiddleAgentOutput(
                    query_id=query_id,
                    predicted_intent=phase1_output.intent,
                    topk_docs=[],
                    retrieval_diagnostics={"latency_ms": 0.0, "strategy_requested": strategy},
                )
            row = evaluate_gold_retrieval_row(
                extracted_retrieved_ids(middle_output),
                gold,
                latency_ms=float(middle_output.retrieval_diagnostics.get("latency_ms", 0.0) or 0.0),
                strategy=strategy,
            )
            eval_rows.append(row)
            all_rows.append(row)

        metric_rows = [
            RetrievalMetricResult(
                precision_at_k=row["precision_at_k"],
                recall_at_k=row["recall_at_k"],
                mrr=row["mrr"],
                latency_ms=row["latency_ms"],
            )
            for row in eval_rows
        ]
        passed = sum(row["passed"] for row in eval_rows)
        per_strategy[strategy] = {
            "gold_rows": len(eval_rows),
            "pass_rate": round(passed / len(eval_rows), 4) if eval_rows else None,
            "summary": {key: round(value, 4) for key, value in summarize_metric_rows(metric_rows).items()},
            "rows": eval_rows,
        }

    overall_metrics = [
        RetrievalMetricResult(
            precision_at_k=row["precision_at_k"],
            recall_at_k=row["recall_at_k"],
            mrr=row["mrr"],
            latency_ms=row["latency_ms"],
        )
        for row in all_rows
    ]
    return {
        "gold_path": str(gold_path),
        "gold_rows": len(gold_rows),
        "strategies": per_strategy,
        "summary": {key: round(value, 4) for key, value in summarize_metric_rows(overall_metrics).items()},
        "retrieval_available": bool(retrieval.available),
        "embedding_model": settings.phase2_embedding_model,
    }


def save_retrieval_gold_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_retrieval_gold_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
