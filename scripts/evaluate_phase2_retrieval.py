from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aviation_rag.config import Settings, ensure_artifact_dirs
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting
from aviation_rag.phase2_metrics import evaluate_ranking, summarize_metric_rows
from aviation_rag.phase2_san_faiss_retrieval import Phase2SanFaissRetrieval

BENCHMARK_QUERIES = [
    {
        "query": "engine warning checklist after takeoff",
        "strategy": "bm25",
        "expected_terms": ["engine", "warning", "checklist"],
    },
    {
        "query": "crosswind turbulence during final approach",
        "strategy": "metadata_first",
        "expected_terms": ["crosswind", "turbulence", "approach"],
    },
    {
        "query": "engine failure after takeoff with emergency return",
        "strategy": "semantic",
        "expected_terms": ["engine", "takeoff", "emergency"],
    },
]


def _doc_is_relevant(doc: dict, terms: list[str]) -> bool:
    text = " ".join(
        [
            doc.get("chunk_text", ""),
            " ".join(str(value) for value in (doc.get("metadata", {}) or {}).values()),
        ]
    ).lower()
    return any(term.lower() in text for term in terms)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    settings = Settings()
    ensure_artifact_dirs(settings)
    phase1 = Phase1HoangIntentRouting(settings)
    retrieval = Phase2SanFaissRetrieval(settings)
    if not retrieval.available:
        raise SystemExit(f"Phase 2 retrieval unavailable: {retrieval.build_error}")

    rows = []
    metric_rows = []
    for case in BENCHMARK_QUERIES:
        phase1_output = phase1.build_output(case["query"], top_k=10, strategy=case["strategy"])
        started = perf_counter()
        output = retrieval.retrieve(phase1_output)
        latency_ms = (perf_counter() - started) * 1000.0
        docs = [doc.model_dump() for doc in output.topk_docs]
        relevant_ids = [doc["doc_id"] for doc in docs if _doc_is_relevant(doc, case["expected_terms"])]
        retrieved_ids = [doc["doc_id"] for doc in docs]
        metric = evaluate_ranking(retrieved_ids, relevant_ids, k=min(5, len(retrieved_ids)), latency_ms=latency_ms)
        metric_rows.append(metric)
        rows.append(
            {
                "query": case["query"],
                "strategy": case["strategy"],
                "metrics": metric.__dict__,
                "diagnostics": output.retrieval_diagnostics,
                "top_docs": docs[:5],
            }
        )

    report = {"summary": summarize_metric_rows(metric_rows), "runs": rows}
    target = settings.artifacts_dir / "phase2_retrieval_benchmark.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(target), "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
