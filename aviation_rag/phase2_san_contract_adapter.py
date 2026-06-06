"""
phase2_san_contract_adapter.py — Phase 2 Retrieval (Quan San)
==============================================================
Adapter connecting the Semantic Retrieval Engine to the Agentic RAG pipeline.

Resolution priority:
    1. Cached output file (if exists for this query_id)
    2. Real retrieval via RetrievalEngine (if index is built)
    3. Sample artifact file (if exists)
    4. Mock fallback (if nothing else is available)

This ensures the pipeline NEVER crashes — even if the index hasn't been
built yet, it falls back gracefully to mock data.

Author: Quan San — Phase 2 Semantic Retrieval Research
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import Settings
from .io_utils import append_jsonl, find_by_query_id, read_jsonl
from .schemas import InputAgentOutput, MiddleAgentOutput, RetrievedDoc

logger = logging.getLogger(__name__)

# ==============================================================================
# Mock Library (fallback when index is not built)
# ==============================================================================

MOCK_RETRIEVAL_LIBRARY: Dict[str, List[dict[str, object]]] = {
    "Incident_Report": [
        {
            "doc_id": "mock_incident_001",
            "chunk_id": "mock_incident_001#0",
            "chunk_text": (
                "Incident-style retrieval placeholder. This chunk simulates a narrative safety report "
                "that would normally come from San's retrieval engine."
            ),
            "scores": {"semantic": 0.92, "hybrid": 0.88, "bm25": 0.31, "final": 0.88},
            "metadata": {"source": "phase2_mock", "document_type": "incident_report"},
        },
        {
            "doc_id": "mock_incident_002",
            "chunk_id": "mock_incident_002#0",
            "chunk_text": (
                "Second incident placeholder chunk with event sequence and corrective actions for grounded QA."
            ),
            "scores": {"semantic": 0.86, "hybrid": 0.82, "bm25": 0.29, "final": 0.82},
            "metadata": {"source": "phase2_mock", "document_type": "incident_report"},
        },
    ],
    "Technical_Procedure": [
        {
            "doc_id": "mock_procedure_001",
            "chunk_id": "mock_procedure_001#0",
            "chunk_text": (
                "Technical procedure placeholder. This chunk represents checklist, QRH, or maintenance steps "
                "that San's retrieval engine would normally return."
            ),
            "scores": {"semantic": 0.63, "hybrid": 0.79, "bm25": 0.91, "final": 0.91},
            "metadata": {"source": "phase2_mock", "document_type": "procedure"},
        },
        {
            "doc_id": "mock_procedure_002",
            "chunk_id": "mock_procedure_002#0",
            "chunk_text": "Backup troubleshooting chunk with maintenance terminology and ordered action steps.",
            "scores": {"semantic": 0.58, "hybrid": 0.75, "bm25": 0.84, "final": 0.84},
            "metadata": {"source": "phase2_mock", "document_type": "procedure"},
        },
    ],
    "Metadata_Query": [
        {
            "doc_id": "mock_metadata_001",
            "chunk_id": "mock_metadata_001#0",
            "chunk_text": (
                "Metadata-style placeholder chunk covering weather, airport, runway, and operating conditions."
            ),
            "scores": {"semantic": 0.55, "hybrid": 0.70, "bm25": 0.78, "final": 0.78},
            "metadata": {"source": "phase2_mock", "document_type": "metadata"},
        },
        {
            "doc_id": "mock_metadata_002",
            "chunk_id": "mock_metadata_002#0",
            "chunk_text": "Additional metadata placeholder with structured fields for filtering and context lookup.",
            "scores": {"semantic": 0.50, "hybrid": 0.66, "bm25": 0.74, "final": 0.74},
            "metadata": {"source": "phase2_mock", "document_type": "metadata"},
        },
    ],
    "Factoid": [
        {
            "doc_id": "mock_factoid_001",
            "chunk_id": "mock_factoid_001#0",
            "chunk_text": "Factoid placeholder chunk with concise aviation reference facts for short-answer questions.",
            "scores": {"semantic": 0.89, "hybrid": 0.83, "bm25": 0.42, "final": 0.83},
            "metadata": {"source": "phase2_mock", "document_type": "factoid"},
        },
        {
            "doc_id": "mock_factoid_002",
            "chunk_id": "mock_factoid_002#0",
            "chunk_text": "Secondary factoid placeholder chunk to support fallback citation coverage.",
            "scores": {"semantic": 0.82, "hybrid": 0.78, "bm25": 0.36, "final": 0.78},
            "metadata": {"source": "phase2_mock", "document_type": "factoid"},
        },
    ],
}


class Phase2SanContractAdapter:
    """
    Adapter for Phase 2 (Quan San — Semantic Retrieval).

    Connects the RetrievalEngine to the LangGraph pipeline.
    Falls back gracefully: real retrieval → sample artifact → mock data.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine = None  # Lazy-loaded

    # ──────────────────────────────────────────────────────────────────
    # Lazy-load Retrieval Engine
    # ──────────────────────────────────────────────────────────────────

    def _ensure_engine(self):
        """Lazy-load the retrieval engine (only when real retrieval is needed)."""
        if self._engine is not None:
            return self._engine

        try:
            from .retrieval.engine import RetrievalEngine
            engine = RetrievalEngine(self.settings)
            if engine.is_available:
                self._engine = engine
                return self._engine
        except Exception as e:
            logger.warning(f"[phase2] Could not load retrieval engine: {e}")

        return None

    # ──────────────────────────────────────────────────────────────────
    # Resolution Methods
    # ──────────────────────────────────────────────────────────────────

    def _pick_sample_row(self, input_row: InputAgentOutput, sample_path: Path) -> MiddleAgentOutput | None:
        if not sample_path.exists():
            return None
        rows = read_jsonl(sample_path)
        if not rows:
            return None

        matching_row = None
        for row in rows:
            if row.get("predicted_intent") == input_row.intent:
                matching_row = row
                break
        if matching_row is None:
            matching_row = rows[0]

        topk_docs = [
            RetrievedDoc.model_validate(doc)
            for doc in matching_row.get("topk_docs", [])
        ]
        return MiddleAgentOutput(
            query_id=input_row.query_id,
            predicted_intent=input_row.intent,
            topk_docs=topk_docs,
            retrieval_diagnostics={
                "adapter_mode": "sample_artifact",
                "contract_owner": "Quan San",
                "strategy_requested": input_row.retrieval_plan.strategy,
            },
        )

    def _build_mock_output(self, input_row: InputAgentOutput) -> MiddleAgentOutput:
        base_docs = MOCK_RETRIEVAL_LIBRARY.get(input_row.intent, MOCK_RETRIEVAL_LIBRARY["Incident_Report"])
        topk_docs: List[RetrievedDoc] = []
        for raw_doc in base_docs[: input_row.retrieval_plan.top_k]:
            doc_payload = dict(raw_doc)
            doc_payload["chunk_text"] = (
                f"Query: {input_row.query_raw}. "
                f"Intent: {input_row.intent}. "
                f"Strategy requested by Hoang phase 1: {input_row.retrieval_plan.strategy}. "
                f"{raw_doc['chunk_text']}"
            )
            topk_docs.append(RetrievedDoc.model_validate(doc_payload))

        return MiddleAgentOutput(
            query_id=input_row.query_id,
            predicted_intent=input_row.intent,
            topk_docs=topk_docs,
            retrieval_diagnostics={
                "adapter_mode": "generated_mock",
                "contract_owner": "Quan San",
                "strategy_requested": input_row.retrieval_plan.strategy,
                "fallback_strategy": input_row.retrieval_plan.fallback_strategy,
                "routing_reason": input_row.retrieval_plan.routing_reason,
                "note": "Index not built. Run: python scripts/build_phase2_san_index.py",
            },
        )

    def _real_retrieval(self, input_row: InputAgentOutput) -> MiddleAgentOutput | None:
        """Attempt real retrieval via the engine."""
        engine = self._ensure_engine()
        if engine is None:
            return None

        try:
            return engine.retrieve(input_row)
        except Exception as e:
            logger.warning(f"[phase2] Real retrieval failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────
    # Main Resolution (priority chain)
    # ──────────────────────────────────────────────────────────────────

    def resolve_output(
        self,
        input_row: InputAgentOutput,
        output_path: Path | None = None,
        sample_path: Path | None = None,
    ) -> MiddleAgentOutput:
        """
        Resolve Phase 2 output with graceful fallback chain:

            1. Cached output (output_path) → return if found
            2. Real retrieval (engine) → return if index exists
            3. Sample artifact (sample_path) → return if found
            4. Mock fallback → always works
        """
        target = output_path or self.settings.phase2_output_path

        # Priority 1: Cached output file
        if target.exists():
            try:
                row = find_by_query_id(target, input_row.query_id)
                return MiddleAgentOutput.model_validate(row)
            except Exception:
                pass

        # Priority 2: Real retrieval via engine
        real_output = self._real_retrieval(input_row)
        if real_output is not None:
            return real_output

        # Priority 3: Sample artifact
        sample_target = sample_path or self.settings.phase2_sample_output_path
        sample_output = self._pick_sample_row(input_row, sample_target)
        if sample_output is not None:
            return sample_output

        # Priority 4: Mock fallback
        return self._build_mock_output(input_row)

    def write_output(self, output: MiddleAgentOutput, path: Path | None = None) -> Path:
        target = path or self.settings.phase2_output_path
        append_jsonl(target, output)
        return target
