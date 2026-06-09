from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .runtime import build_run_state

if TYPE_CHECKING:
    from .config import Settings


RetrievalStrategyInput = Literal["bm25", "semantic", "hybrid", "metadata_first", "hybrid_rrf"]


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    strategy: RetrievalStrategyInput | None = None
    allow_local_fallback: bool = True
    write_phase1_artifact: bool = True


class ChatResponse(BaseModel):
    query_id: str
    intent: str | None = None
    intent_source: str | None = None
    answer: str
    topk_docs: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    hallucination_risk: float | None = None
    retrieval_diagnostics: dict[str, Any]
    phase3_artifact_path: str | None = None


_runtime_lock = Lock()


def _ensure_runtime(app: FastAPI) -> tuple[Any, Any]:
    settings = getattr(app.state, "settings", None)
    graph_app = getattr(app.state, "graph_app", None)
    if settings is not None and graph_app is not None:
        return settings, graph_app

    with _runtime_lock:
        settings = getattr(app.state, "settings", None)
        graph_app = getattr(app.state, "graph_app", None)
        if settings is None or graph_app is None:
            from .config import Settings, configure_tracing_env, ensure_artifact_dirs
            from .graph import build_graph

            settings = settings or Settings()
            ensure_artifact_dirs(settings)
            configure_tracing_env(settings)
            graph_app = build_graph(settings)
            app.state.settings = settings
            app.state.graph_app = graph_app
    return app.state.settings, app.state.graph_app


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="Hoang Intent-Aware Aviation Workflow API",
        version="1.1.0",
        description="Phase 1 intent routing, complete Phase 2 dense/BM25 retrieval, and Phase 3 grounded QA.",
    )

    if settings is not None:
        app.state.settings = settings
        app.state.graph_app = None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            runtime_settings, graph_app = _ensure_runtime(app)
            state = build_run_state(
                runtime_settings,
                query_raw=request.query,
                top_k=request.top_k,
                strategy=request.strategy,
                allow_local_fallback=request.allow_local_fallback,
                write_phase1_artifact=request.write_phase1_artifact,
                write_phase2_artifact=True,
                write_phase3_artifact=True,
            )
            result = graph_app.invoke(state)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc

        return ChatResponse(
            query_id=str(result.get("query_id", "")),
            intent=result.get("intent"),
            intent_source=result.get("intent_source"),
            answer=str(result.get("answer", "")),
            topk_docs=result.get("topk_docs", []) or [],
            citations=result.get("citations", []) or [],
            hallucination_risk=result.get("hallucination_risk"),
            retrieval_diagnostics=result.get("retrieval_diagnostics", {}) or {},
            phase3_artifact_path=result.get("phase3_artifact_path"),
        )

    return app


app = create_app()
