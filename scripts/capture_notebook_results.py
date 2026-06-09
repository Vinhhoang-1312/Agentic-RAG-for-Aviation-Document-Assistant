"""Capture detailed notebook outputs for Phase 1 and Phase 3."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from aviation_rag.config import Settings, ensure_artifact_dirs
from aviation_rag.io_utils import read_jsonl, write_jsonl
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting, heuristic_intent
from aviation_rag.phase2_san_contract_adapter import MOCK_RETRIEVAL_LIBRARY, Phase2SanContractAdapter
from aviation_rag.phase2_san_faiss_retrieval import Phase2SanFaissRetrieval
from aviation_rag.phase3_hoang_grounded_qa import Phase3HoangGroundedQA
from aviation_rag.schemas import InputAgentOutput

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------- PHASE 1 ----------
section("PHASE 1 - Cell 1 Setup")
base_settings = Settings()
settings = replace(base_settings, input_intent_mode="ml")
ensure_artifact_dirs(settings)
phase1_preview = Phase1HoangIntentRouting(settings)
print(f"Intent runtime mode: {settings.input_intent_mode}")
print(f"Training mode: {phase1_preview.intent_training_mode}")
print(f"Training rows: {phase1_preview.intent_model.training_rows}")
print(f"Label counts: {phase1_preview.intent_model.label_counts}")

sample_queries = [
    {"query_id": "q_incident_001", "query_raw": "engine failure after takeoff with emergency return", "expected_intent": "Incident_Report", "expected_strategy": "semantic"},
    {"query_id": "q_procedure_001", "query_raw": "den bao ENG OIL PRESS sang thi lam gi?", "expected_intent": "Technical_Procedure", "expected_strategy": "bm25"},
    {"query_id": "q_metadata_001", "query_raw": "crosswind turbulence during final approach at runway 25", "expected_intent": "Metadata_Query", "expected_strategy": "metadata_first"},
    {"query_id": "q_factoid_001", "query_raw": "what is the meaning of MEL in aviation?", "expected_intent": "Factoid", "expected_strategy": "semantic"},
]

section("PHASE 1 - Cell 6 ML vs Heuristic")
phase1 = Phase1HoangIntentRouting(settings)
phase1_rows = []
comparison_rows = []
for item in sample_queries:
    output = phase1.build_output(query_raw=item["query_raw"], query_id=item["query_id"], top_k=10)
    phase1_rows.append(output)
    baseline_intent = heuristic_intent(output.query_normalized)
    comparison_rows.append({
        "query_id": output.query_id,
        "expected": item["expected_intent"],
        "ml_intent": output.intent,
        "ml_source": output.intent_source,
        "ml_confidence": round(float(output.intent_confidence), 4),
        "heuristic_intent": baseline_intent,
        "ml_ok": output.intent == item["expected_intent"],
        "heuristic_ok": baseline_intent == item["expected_intent"],
    })
comparison_frame = pd.DataFrame(comparison_rows)
summary_frame = pd.DataFrame([
    {"method": "TF-IDF + Logistic Regression", "intent_accuracy": comparison_frame["ml_ok"].mean(), "confidence_mean": comparison_frame["ml_confidence"].mean()},
    {"method": "Heuristic baseline", "intent_accuracy": comparison_frame["heuristic_ok"].mean(), "confidence_mean": 0.60},
])
print(summary_frame.to_string(index=False))
print("\nPer query:")
print(comparison_frame.to_string(index=False))

section("PHASE 1 - Cell 8 Routing")
routing_frame = pd.DataFrame([
    {"query_id": r.query_id, "intent": r.intent, "strategy": r.retrieval_plan.strategy, "fallback": r.retrieval_plan.fallback_strategy}
    for r in phase1_rows
])
print(routing_frame.to_string(index=False))

section("PHASE 1 - Cell 10 BM25 vs Semantic")
retrieval_settings = replace(
    settings,
    phase2_embedding_model="tfidf_svd_fallback",
    phase2_index_dir=settings.artifacts_dir / "phase2_index_fast",
    retrieval_max_docs=min(settings.retrieval_max_docs, 6000),
    retrieval_svd_components=min(settings.retrieval_svd_components, 96),
)
ensure_artifact_dirs(retrieval_settings)
bm25_probe_query = "qrh checklist for engine fire warning"
probe_phase1 = Phase1HoangIntentRouting(retrieval_settings)
retrieval = Phase2SanFaissRetrieval(retrieval_settings)
strategy_outputs = []
detail_rows = []
for strategy in ["bm25", "semantic"]:
    input_row = probe_phase1.build_output(bm25_probe_query, query_id=f"probe_{strategy}", top_k=5, strategy=strategy)
    output = retrieval.retrieve(input_row)
    top_docs = output.topk_docs[:3]
    top_doc = top_docs[0]
    strategy_outputs.append({
        "strategy": strategy,
        "top_doc_id": top_doc.doc_id,
        "top_doc_type": top_doc.metadata.get("document_type"),
        "semantic": round(float(top_doc.scores.get("semantic", 0)), 4),
        "bm25": round(float(top_doc.scores.get("bm25", 0)), 4),
        "metadata": round(float(top_doc.scores.get("metadata", 0)), 4),
        "final": round(float(top_doc.scores.get("final", 0)), 4),
        "weights": output.retrieval_diagnostics.get("score_weights"),
    })
    for rank, doc in enumerate(top_docs, 1):
        detail_rows.append({"strategy": strategy, "rank": rank, "doc_id": doc.doc_id, "final": round(float(doc.scores.get("final", 0)), 4)})
bm25_df = pd.DataFrame(strategy_outputs)
print("Top-1 comparison:")
print(bm25_df.to_string(index=False))
print(f"\ntop_doc_changed: {bm25_df.loc[0,'top_doc_id'] != bm25_df.loc[1,'top_doc_id']}")
print("\nTop-3 detail:")
print(pd.DataFrame(detail_rows).to_string(index=False))

section("PHASE 1 - Cell 14 Evaluation")
expected = {item["query_id"]: item for item in sample_queries}
eval_rows = []
for row in phase1_rows:
    gold = expected[row.query_id]
    eval_rows.append({
        "query_id": row.query_id,
        "predicted_intent": row.intent,
        "expected_intent": gold["expected_intent"],
        "intent_ok": row.intent == gold["expected_intent"],
        "predicted_strategy": row.retrieval_plan.strategy,
        "expected_strategy": gold["expected_strategy"],
        "strategy_ok": row.retrieval_plan.strategy == gold["expected_strategy"],
        "confidence": round(float(row.intent_confidence), 4),
    })
eval_df = pd.DataFrame(eval_rows)
print(eval_df.to_string(index=False))
print(f"\nSummary: intent_accuracy={eval_df['intent_ok'].mean():.2%}, routing_accuracy={eval_df['strategy_ok'].mean():.2%}")

# ---------- PHASE 3 ----------
section("PHASE 3 - Cell 3 Load Phase 1")
settings3 = Settings()
phase1_rows_json = read_jsonl(settings3.phase1_output_path)
print(f"Loaded {len(phase1_rows_json)} Phase 1 rows")
print(f"First query_id: {phase1_rows_json[0]['query_id']}, intent: {phase1_rows_json[0]['intent']}, source: {phase1_rows_json[0]['intent_source']}")

section("PHASE 3 - Cell 5 Mock inventory")
print(f"Mock intents: {list(MOCK_RETRIEVAL_LIBRARY.keys())}")
print(f"Total mock chunks: {sum(len(v) for v in MOCK_RETRIEVAL_LIBRARY.values())}")

section("PHASE 3 - Cell 7 Resolve Phase 2")
phase2_adapter = Phase2SanContractAdapter(settings3)
phase2_rows = []
for row in phase1_rows_json:
    phase1_output = InputAgentOutput.model_validate(row)
    phase2_rows.append(phase2_adapter.resolve_output(phase1_output))
phase2_summary = pd.DataFrame([
    {
        "query_id": r.query_id,
        "intent": r.predicted_intent,
        "adapter_mode": r.retrieval_diagnostics.get("adapter_mode"),
        "topk_docs": len(r.topk_docs),
        "first_doc_id": r.topk_docs[0].doc_id if r.topk_docs else None,
        "first_doc_source": r.topk_docs[0].metadata.get("source") if r.topk_docs else None,
    }
    for r in phase2_rows
])
print(phase2_summary.to_string(index=False))

section("PHASE 3 - Cell 9 Generate answers")
phase3 = Phase3HoangGroundedQA(settings3)
phase1_lookup = {row["query_id"]: row for row in phase1_rows_json}
phase3_rows = []
for phase2_output in phase2_rows:
    query_raw = phase1_lookup[phase2_output.query_id]["query_raw"]
    phase3_rows.append(phase3.generate(question=query_raw, middle_output=phase2_output, allow_fallback=True, force_local=True))
phase3_preview = pd.DataFrame([
    {"query_id": r.query_id, "answer_preview": r.answer[:120], "citations": len(r.citations), "hallucination_risk": round(float(r.hallucination_risk), 4)}
    for r in phase3_rows
])
print(phase3_preview.to_string(index=False))

section("PHASE 3 - Cell 13 Hallucination debug")
import re

def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

risk_rows = []
for p3, p2 in zip(phase3_rows, phase2_rows):
    answer_tokens = token_set(p3.answer)
    context_tokens = token_set(" ".join(d.chunk_text for d in p2.topk_docs))
    overlap = answer_tokens & context_tokens
    risk_rows.append({
        "query_id": p3.query_id,
        "answer_tokens": len(answer_tokens),
        "overlap_tokens": len(overlap),
        "overlap_ratio": round(len(overlap) / max(1, len(answer_tokens)), 4),
        "hallucination_risk": round(float(p3.hallucination_risk), 4),
    })
print(pd.DataFrame(risk_rows).to_string(index=False))

section("PHASE 3 - Cell 15 Quality metrics")
quality = pd.DataFrame([
    {"query_id": r.query_id, "citations": len(r.citations), "has_citation": bool(r.citations), "risk": round(float(r.hallucination_risk), 4), "empty": not r.answer.strip()}
    for r in phase3_rows
])
print(quality.to_string(index=False))
print(f"\nSummary: citation_coverage={quality['has_citation'].mean():.0%}, avg_citations={quality['citations'].mean():.1f}, avg_risk={quality['risk'].mean():.3f}, empty_rate={quality['empty'].mean():.0%}")
