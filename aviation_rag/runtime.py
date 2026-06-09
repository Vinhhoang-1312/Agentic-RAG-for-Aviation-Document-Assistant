from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings
from .phase2_san_contract_adapter import Phase2SanContractAdapter
from .phase2_san_faiss_retrieval import Phase2SanFaissRetrieval
from .schemas import InputAgentOutput, MiddleAgentOutput

RUN_MODES = ("Fast local", "Full dense/Route LLM")


def notebook_settings(base: Settings | None = None, *, run_mode: str = "Fast local") -> Settings:
    """Mirror Streamlit runtime settings for notebooks."""
    from dataclasses import replace

    settings = base or Settings()
    if run_mode == "Fast local":
        return replace(
            settings,
            input_intent_mode="auto",
            phase2_embedding_model="tfidf_svd_fallback",
            phase2_index_dir=settings.artifacts_dir / "phase2_index_fast",
            retrieval_max_docs=min(settings.retrieval_max_docs, 6000),
            retrieval_svd_components=min(settings.retrieval_svd_components, 96),
        )
    return replace(settings, input_intent_mode="auto")


def force_local_for_run_mode(run_mode: str) -> bool:
    return run_mode == "Fast local"


def resolve_phase2_output(
    settings: Settings,
    phase1_output: InputAgentOutput,
    *,
    phase2_retrieval: Phase2SanFaissRetrieval | None = None,
    phase2_adapter: Phase2SanContractAdapter | None = None,
    artifact_path: Path | None = None,
) -> MiddleAgentOutput:
    """Same Phase 2 path as Streamlit graph: artifact -> mock -> local FAISS retrieval."""
    adapter = phase2_adapter or Phase2SanContractAdapter(settings)
    retrieval = phase2_retrieval or Phase2SanFaissRetrieval(settings)
    target = artifact_path or settings.phase2_output_path
    phase2_output = adapter.resolve_output(phase1_output, output_path=target)

    if phase2_output.retrieval_diagnostics.get("adapter_mode") == "generated_mock":
        if retrieval.available:
            phase2_output = retrieval.retrieve(phase1_output)
        else:
            phase2_output.retrieval_diagnostics.update(
                {
                    "retrieval_backend": "generated_mock",
                    "embedding_model": settings.phase2_embedding_model,
                    "embedding_backend": "unavailable",
                    "embedding_dim": 0,
                    "faiss_index_type": "IndexFlatIP",
                    "normalization": "L2",
                    "chunk_count": 0,
                    "bm25_enabled": False,
                    "metadata_filter_applied": False,
                    "fusion_method": "none",
                    "latency_ms": 0.0,
                    "fallback_reason": retrieval.build_error,
                }
            )
    return phase2_output


def build_run_state(
    settings: Settings,
    *,
    query_raw: str | None = None,
    query_id: str | None = None,
    top_k: int | None = None,
    strategy: str | None = None,
    allow_local_fallback: bool = True,
    force_local_answer: bool = False,
    write_phase1_artifact: bool = False,
    write_phase2_artifact: bool = True,
    write_phase3_artifact: bool = True,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "allow_local_fallback": bool(allow_local_fallback),
        "force_local_answer": bool(force_local_answer),
        "write_phase1_artifact": bool(write_phase1_artifact),
        "write_phase2_artifact": bool(write_phase2_artifact),
        "write_phase3_artifact": bool(write_phase3_artifact),
    }
    if cancel_event is not None:
        state["cancel_event"] = cancel_event
    if query_raw:
        state["query_raw"] = query_raw
    if query_id:
        state["query_id"] = query_id
    if top_k is not None or strategy is not None:
        state["retrieval_plan_override"] = {
            "top_k": int(top_k or settings.default_top_k),
            "strategy": str(strategy or settings.default_strategy),
        }
    return state
