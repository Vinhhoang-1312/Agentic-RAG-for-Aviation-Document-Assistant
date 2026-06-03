from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetrievalMetricResult:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    latency_ms: float


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if k <= 0 or not retrieved_ids:
        return 0.0
    top = list(retrieved_ids[:k])
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top.intersection(relevant)) / len(relevant)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for index, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def evaluate_ranking(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    k: int,
    latency_ms: float,
) -> RetrievalMetricResult:
    return RetrievalMetricResult(
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        mrr=reciprocal_rank(retrieved_ids, relevant_ids),
        latency_ms=latency_ms,
    )


def timed(call: Callable[[], T]) -> tuple[T, float]:
    started = perf_counter()
    value = call()
    return value, (perf_counter() - started) * 1000.0


def summarize_metric_rows(rows: Sequence[RetrievalMetricResult]) -> dict[str, float]:
    if not rows:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "latency_ms": 0.0}
    return {
        "precision_at_k": mean(row.precision_at_k for row in rows),
        "recall_at_k": mean(row.recall_at_k for row in rows),
        "mrr": mean(row.mrr for row in rows),
        "latency_ms": mean(row.latency_ms for row in rows),
    }
