from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


IntentLabel = Literal[
    "Incident_Report",
    "Technical_Procedure",
    "Metadata_Query",
    "Factoid",
]

IntentSource = Literal["ml", "heuristic"]

RetrievalStrategy = Literal["bm25", "semantic", "hybrid", "metadata_first", "hybrid_rrf"]


class RetrievalPlan(BaseModel):
    strategy: RetrievalStrategy = "hybrid"
    fallback_strategy: RetrievalStrategy = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Dict[str, Any] = Field(default_factory=dict)
    routing_reason: str = ""


class InputAgentOutput(BaseModel):
    query_id: str
    query_raw: str
    query_normalized: str
    intent: IntentLabel
    intent_confidence: float = Field(ge=0.0, le=1.0)
    intent_source: IntentSource = "ml"
    expanded_queries: List[str] = Field(default_factory=list)
    rewritten_query: str
    retrieval_plan: RetrievalPlan
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("expanded_queries")
    @classmethod
    def dedupe_expansions(cls, values: List[str]) -> List[str]:
        unique: List[str] = []
        seen = set()
        for value in values:
            key = value.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(value.strip())
        return unique


class RetrievedDoc(BaseModel):
    doc_id: str
    chunk_id: str
    chunk_text: str
    scores: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MiddleAgentOutput(BaseModel):
    query_id: str
    predicted_intent: IntentLabel
    topk_docs: List[RetrievedDoc]
    retrieval_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Citation(BaseModel):
    doc_id: str
    chunk_id: Optional[str] = None
    reason: str = ""


class FinalOutput(BaseModel):
    query_id: str
    query_raw: str = ""
    answer: str
    citations: List[Citation]
    hallucination_risk: float = Field(ge=0.0, le=1.0)
    grounding_report: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
