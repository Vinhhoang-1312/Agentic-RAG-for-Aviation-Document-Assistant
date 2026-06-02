from __future__ import annotations

from typing import Any

from .config import Settings


def build_run_state(
    settings: Settings,
    *,
    query_raw: str | None = None,
    query_id: str | None = None,
    top_k: int | None = None,
    strategy: str | None = None,
    allow_local_fallback: bool = True,
    write_phase1_artifact: bool = False,
    write_phase2_artifact: bool = True,
    write_phase3_artifact: bool = True,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "allow_local_fallback": bool(allow_local_fallback),
        "write_phase1_artifact": bool(write_phase1_artifact),
        "write_phase2_artifact": bool(write_phase2_artifact),
        "write_phase3_artifact": bool(write_phase3_artifact),
    }
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
