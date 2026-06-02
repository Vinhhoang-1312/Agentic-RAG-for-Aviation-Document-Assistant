from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings, configure_tracing_env
from .io_utils import find_by_query_id
from .phase2_san_faiss_retrieval import Phase2SanFaissRetrieval
from .phase1_hoang_intent_routing import Phase1HoangIntentRouting
from .phase2_san_contract_adapter import Phase2SanContractAdapter
from .phase3_hoang_grounded_qa import Phase3HoangGroundedQA
from .schemas import InputAgentOutput, MiddleAgentOutput


class RagState(TypedDict, total=False):
    query_id: str
    query_raw: str
    query_normalized: str
    intent: str
    intent_confidence: float
    intent_source: str
    expanded_queries: list[str]
    rewritten_query: str
    retrieval_plan: dict[str, Any]
    retrieval_plan_override: dict[str, Any]
    phase1_artifact_path: str
    phase2_artifact_path: str
    phase3_artifact_path: str
    topk_docs: list[dict[str, Any]]
    retrieval_diagnostics: dict[str, Any]
    answer: str
    citations: list[dict[str, Any]]
    hallucination_risk: float
    grounding_report: dict[str, Any]
    allow_local_fallback: bool
    write_phase1_artifact: bool
    write_phase2_artifact: bool
    write_phase3_artifact: bool


def build_graph(settings: Settings):
    configure_tracing_env(settings)
    phase1_agent = Phase1HoangIntentRouting(settings)
    phase2_retrieval = Phase2SanFaissRetrieval(settings)
    phase2_adapter = Phase2SanContractAdapter(settings)
    phase3_agent = Phase3HoangGroundedQA(settings)

    def phase1_hoang_input_node(state: RagState) -> RagState:
        phase1_path = Path(state.get("phase1_artifact_path", str(settings.phase1_output_path)))
        write_phase1 = bool(state.get("write_phase1_artifact", False))

        if "query_id" in state and "query_raw" not in state:
            row = find_by_query_id(phase1_path, state["query_id"])
            phase1_output = InputAgentOutput.model_validate(row)
        elif "query_raw" in state:
            override = state.get("retrieval_plan_override", {})
            phase1_output = phase1_agent.build_output(
                query_raw=state["query_raw"],
                query_id=state.get("query_id"),
                top_k=override.get("top_k", settings.default_top_k),
                strategy=override.get("strategy"),
            )
            if write_phase1:
                phase1_agent.write_output(phase1_output, phase1_path)
        else:
            raise ValueError("State must include `query_raw` or (`query_id` + phase1 artifact).")

        return {
            "query_id": phase1_output.query_id,
            "query_raw": phase1_output.query_raw,
            "query_normalized": phase1_output.query_normalized,
            "intent": phase1_output.intent,
            "intent_confidence": phase1_output.intent_confidence,
            "intent_source": phase1_output.intent_source,
            "expanded_queries": phase1_output.expanded_queries,
            "rewritten_query": phase1_output.rewritten_query,
            "retrieval_plan": phase1_output.retrieval_plan.model_dump(),
            "phase1_artifact_path": str(phase1_path),
        }

    def phase2_san_contract_node(state: RagState) -> RagState:
        phase1_output = InputAgentOutput(
            query_id=state["query_id"],
            query_raw=state["query_raw"],
            query_normalized=state["query_normalized"],
            intent=state["intent"],
            intent_confidence=float(state["intent_confidence"]),
            intent_source=state.get("intent_source", "heuristic"),
            expanded_queries=state.get("expanded_queries", []),
            rewritten_query=state["rewritten_query"],
            retrieval_plan=state["retrieval_plan"],
        )
        phase2_path = Path(state.get("phase2_artifact_path", str(settings.phase2_output_path)))
        phase2_output = phase2_adapter.resolve_output(
            phase1_output,
            output_path=phase2_path,
        )
        if phase2_retrieval.available and phase2_output.retrieval_diagnostics.get("adapter_mode") == "generated_mock":
            phase2_output = phase2_retrieval.retrieve(phase1_output)
        if bool(state.get("write_phase2_artifact", True)):
            phase2_adapter.write_output(phase2_output, phase2_path)
        return {
            "topk_docs": [doc.model_dump() for doc in phase2_output.topk_docs],
            "retrieval_diagnostics": phase2_output.retrieval_diagnostics,
            "phase2_artifact_path": str(phase2_path),
        }

    def phase3_hoang_output_node(state: RagState) -> RagState:
        phase2_output = MiddleAgentOutput(
            query_id=state["query_id"],
            predicted_intent=state["intent"],
            topk_docs=state["topk_docs"],
            retrieval_diagnostics=state.get("retrieval_diagnostics", {}),
        )
        final_output = phase3_agent.generate(
            question=state["query_raw"],
            middle_output=phase2_output,
            allow_fallback=bool(state.get("allow_local_fallback", True)),
        )
        phase3_path = Path(state.get("phase3_artifact_path", str(settings.phase3_output_path)))
        if bool(state.get("write_phase3_artifact", True)):
            phase3_agent.write_output(final_output, phase3_path)
        return {
            "answer": final_output.answer,
            "citations": [item.model_dump() for item in final_output.citations],
            "hallucination_risk": final_output.hallucination_risk,
            "grounding_report": final_output.grounding_report,
            "phase3_artifact_path": str(phase3_path),
        }

    graph = StateGraph(RagState)
    graph.add_node("phase1_hoang_input_node", phase1_hoang_input_node)
    graph.add_node("phase2_san_contract_node", phase2_san_contract_node)
    graph.add_node("phase3_hoang_output_node", phase3_hoang_output_node)
    graph.add_edge(START, "phase1_hoang_input_node")
    graph.add_edge("phase1_hoang_input_node", "phase2_san_contract_node")
    graph.add_edge("phase2_san_contract_node", "phase3_hoang_output_node")
    graph.add_edge("phase3_hoang_output_node", END)
    return graph.compile()
