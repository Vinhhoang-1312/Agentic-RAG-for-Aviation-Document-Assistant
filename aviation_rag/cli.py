from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings, configure_tracing_env, ensure_artifact_dirs
from .graph import build_graph
from .runtime import build_run_state


STRATEGIES = ["bm25", "semantic", "hybrid", "metadata_first", "hybrid_rrf"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Hoang's phase-based LangGraph workflow.")
    parser.add_argument("--query", type=str, default=None, help="Raw user query.")
    parser.add_argument("--query-id", type=str, default=None, help="Existing query_id in phase 1 artifact.")
    parser.add_argument("--phase1-artifact", type=str, default=None, help="Path to phase1_hoang_intent_routing_output.jsonl")
    parser.add_argument("--phase2-artifact", type=str, default=None, help="Path to phase2_san_retrieval_output.jsonl")
    parser.add_argument("--phase3-artifact", type=str, default=None, help="Path to phase3_hoang_grounded_answer_output.jsonl")
    parser.add_argument("--top-k", type=int, default=None, help="Override routing top-k.")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=STRATEGIES,
        help="Manual override for retrieval strategy.",
    )
    parser.add_argument(
        "--write-phase1-artifact",
        action="store_true",
        help="When --query is provided, write Hoang phase 1 output artifact.",
    )
    parser.add_argument(
        "--no-local-fallback",
        action="store_true",
        help="Disable phase 3 fallback when Route LLM/OpenRouter is unavailable.",
    )
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = build_parser()
    args = parser.parse_args()
    if not args.query and not args.query_id:
        parser.error("Provide --query or --query-id.")

    settings = Settings()
    ensure_artifact_dirs(settings)
    configure_tracing_env(settings)
    app = build_graph(settings)

    state = build_run_state(
        settings,
        query_raw=args.query,
        query_id=args.query_id,
        top_k=args.top_k,
        strategy=args.strategy,
        allow_local_fallback=not args.no_local_fallback,
        write_phase1_artifact=bool(args.write_phase1_artifact),
        write_phase2_artifact=True,
        write_phase3_artifact=True,
    )
    if args.phase1_artifact:
        state["phase1_artifact_path"] = str(Path(args.phase1_artifact))
    if args.phase2_artifact:
        state["phase2_artifact_path"] = str(Path(args.phase2_artifact))
    if args.phase3_artifact:
        state["phase3_artifact_path"] = str(Path(args.phase3_artifact))

    result = app.invoke(state)
    output = {
        "query_id": result.get("query_id"),
        "intent": result.get("intent"),
        "intent_source": result.get("intent_source"),
        "retrieval_diagnostics": result.get("retrieval_diagnostics", {}),
        "hallucination_risk": result.get("hallucination_risk"),
        "answer_preview": (result.get("answer", "")[:280] + "...") if result.get("answer") else "",
        "phase3_artifact_path": result.get("phase3_artifact_path", str(settings.phase3_output_path)),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
