from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import Settings, configure_tracing_env, ensure_artifact_dirs
from .graph import build_graph
from .io_utils import append_jsonl
from .runtime import build_run_state


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive CLI over Hoang's phase-based workflow.")
    parser.add_argument("--session-id", type=str, default=None, help="Optional session id.")
    parser.add_argument("--max-turns", type=int, default=0, help="Stop automatically after N turns.")
    parser.add_argument("--top-k", type=int, default=None, help="Override routing top-k.")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["bm25", "semantic", "hybrid", "metadata_first"],
        help="Manual strategy override.",
    )
    parser.add_argument(
        "--no-local-fallback",
        action="store_true",
        help="Disable phase 3 fallback when OpenAI is unavailable.",
    )
    return parser


def _session_path(settings: Settings, session_id: str) -> Path:
    path = settings.artifacts_dir / "chat_sessions" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _print_assistant(result: dict[str, Any]) -> None:
    print("\nAssistant:")
    print((result.get("answer") or "").strip() or "[empty answer]")
    print("\nDiagnostics:")
    print(f"- intent: {result.get('intent', 'unknown')}")
    print(f"- intent_source: {result.get('intent_source', 'unknown')}")
    if result.get("hallucination_risk") is not None:
        print(f"- hallucination_risk: {float(result['hallucination_risk']):.4f}")
    citations = result.get("citations", []) or []
    if citations:
        print("- citations: " + ", ".join(c.get("doc_id", "") for c in citations[:5] if c.get("doc_id")))
    else:
        print("- citations: none")
    print()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = build_parser().parse_args()
    settings = Settings()
    ensure_artifact_dirs(settings)
    configure_tracing_env(settings)
    app = build_graph(settings)

    session_id = args.session_id or f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    log_path = _session_path(settings, session_id)

    print("Hoang phase-based chat is ready.")
    print("Commands: /exit, /quit, /help")
    print(f"Session: {session_id}")

    turn = 0
    while True:
        if args.max_turns > 0 and turn >= args.max_turns:
            print("Reached max turns. Session ended.")
            break

        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            print("Session ended.")
            break
        if user_text.lower() == "/help":
            print("Commands: /exit, /quit, /help")
            continue

        turn += 1
        state = build_run_state(
            settings,
            query_raw=user_text,
            top_k=args.top_k,
            strategy=args.strategy,
            allow_local_fallback=not args.no_local_fallback,
            write_phase1_artifact=True,
            write_phase2_artifact=True,
            write_phase3_artifact=True,
        )
        result = app.invoke(state)
        _print_assistant(result)

        append_jsonl(
            log_path,
            {
                "session_id": session_id,
                "turn_index": turn,
                "timestamp": _now_iso(),
                "user_query": user_text,
                "query_id": result.get("query_id"),
                "intent": result.get("intent"),
                "intent_source": result.get("intent_source"),
                "answer": result.get("answer"),
                "citations": result.get("citations", []),
                "hallucination_risk": result.get("hallucination_risk"),
                "retrieval_diagnostics": result.get("retrieval_diagnostics", {}),
            },
        )

    summary = {
        "session_id": session_id,
        "chat_log_path": str(log_path),
        "phase1_output_path": str(settings.phase1_output_path),
        "phase2_output_path": str(settings.phase2_output_path),
        "phase3_output_path": str(settings.phase3_output_path),
        "turns": turn,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
