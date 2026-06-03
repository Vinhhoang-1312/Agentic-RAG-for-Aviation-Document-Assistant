from __future__ import annotations

import unittest

from aviation_rag.schemas import FinalOutput, InputAgentOutput, MiddleAgentOutput, RetrievalPlan, RetrievedDoc


class ContractTests(unittest.TestCase):
    def test_phase1_contract(self):
        output = InputAgentOutput(
            query_id="q_0001",
            query_raw="engine warning during climb",
            query_normalized="engine warning during climb",
            intent="Technical_Procedure",
            intent_confidence=0.88,
            intent_source="ml",
            expanded_queries=["engine warning", "checklist for engine warning"],
            rewritten_query="aviation troubleshooting and procedure lookup for: engine warning during climb",
            retrieval_plan=RetrievalPlan(
                strategy="bm25",
                fallback_strategy="hybrid",
                top_k=10,
                filters={"document_type": "procedure"},
                routing_reason="Procedure-style queries favor checklist retrieval.",
            ),
        )
        self.assertEqual(output.intent_source, "ml")
        self.assertEqual(output.retrieval_plan.fallback_strategy, "hybrid")

    def test_phase2_contract(self):
        output = MiddleAgentOutput(
            query_id="q_0002",
            predicted_intent="Incident_Report",
            topk_docs=[
                RetrievedDoc(
                    doc_id="doc_1001",
                    chunk_id="doc_1001#0",
                    chunk_text="sample retrieval chunk",
                    scores={"bm25": 0.2, "semantic": 0.7, "metadata": 0.1, "final": 0.5},
                    metadata={"source": "phase2_contract"},
                )
            ],
            retrieval_diagnostics={"adapter_mode": "faiss_retrieval", "fusion_method": "weighted_linear_fusion"},
        )
        self.assertEqual(len(output.topk_docs), 1)

    def test_phase2_hybrid_rrf_strategy_is_valid_contract(self):
        plan = RetrievalPlan(strategy="hybrid_rrf", fallback_strategy="hybrid", top_k=5)
        self.assertEqual(plan.strategy, "hybrid_rrf")

    def test_phase3_contract(self):
        output = FinalOutput(
            query_id="q_0003",
            answer="Use checklist ABC and notify ATC.",
            citations=[{"doc_id": "doc_1001", "reason": "matches warning procedure"}],
            hallucination_risk=0.22,
            grounding_report={"overlap_ratio": 0.78},
        )
        self.assertTrue(output.answer.startswith("Use checklist"))


if __name__ == "__main__":
    unittest.main()
