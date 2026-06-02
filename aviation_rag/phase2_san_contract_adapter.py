from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .config import Settings
from .io_utils import append_jsonl, find_by_query_id
from .schemas import InputAgentOutput, MiddleAgentOutput, RetrievedDoc

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
    def __init__(self, settings: Settings):
        self.settings = settings

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
                "contract_owner": "Quang San",
                "strategy_requested": input_row.retrieval_plan.strategy,
                "fallback_strategy": input_row.retrieval_plan.fallback_strategy,
                "routing_reason": input_row.retrieval_plan.routing_reason,
            },
        )

    def resolve_output(
        self,
        input_row: InputAgentOutput,
        output_path: Path | None = None,
    ) -> MiddleAgentOutput:
        target = output_path or self.settings.phase2_output_path
        if target.exists():
            try:
                row = find_by_query_id(target, input_row.query_id)
                return MiddleAgentOutput.model_validate(row)
            except Exception:
                pass

        return self._build_mock_output(input_row)

    def write_output(self, output: MiddleAgentOutput, path: Path | None = None) -> Path:
        target = path or self.settings.phase2_output_path
        append_jsonl(target, output)
        return target
