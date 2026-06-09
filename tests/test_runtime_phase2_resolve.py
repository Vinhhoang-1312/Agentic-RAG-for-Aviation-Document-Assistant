from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import MagicMock

from aviation_rag.config import Settings
from aviation_rag.runtime import force_local_for_run_mode, notebook_settings, resolve_phase2_output
from aviation_rag.schemas import InputAgentOutput, MiddleAgentOutput, RetrievalPlan, RetrievedDoc


def _phase1_output() -> InputAgentOutput:
    return InputAgentOutput(
        query_id="q_test",
        query_raw="engine failure after takeoff",
        query_normalized="engine failure after takeoff",
        intent="Incident_Report",
        intent_confidence=0.9,
        intent_source="ml",
        expanded_queries=["engine failure after takeoff"],
        rewritten_query="aviation incident narrative lookup",
        retrieval_plan=RetrievalPlan(strategy="semantic", fallback_strategy="hybrid", top_k=3),
    )


class RuntimePhase2ResolveTests(unittest.TestCase):
    def test_force_local_matches_streamlit_fast_mode(self):
        self.assertTrue(force_local_for_run_mode("Fast local"))
        self.assertFalse(force_local_for_run_mode("Full dense/Route LLM"))

    def test_notebook_settings_fast_local_uses_fallback_index(self):
        base = Settings()
        settings = notebook_settings(base, run_mode="Fast local")
        self.assertEqual(settings.phase2_embedding_model, "tfidf_svd_fallback")
        self.assertTrue(str(settings.phase2_index_dir).endswith("phase2_index_fast"))

    def test_resolve_phase2_uses_local_retrieval_when_mock(self):
        phase1_output = _phase1_output()
        mock_doc = RetrievedDoc(
            doc_id="747383",
            chunk_id="747383#0",
            chunk_text="ASRS incident narrative",
            scores={"semantic": 0.9, "bm25": 0.4, "metadata": 0.1, "final": 0.9},
            metadata={"source": "asrs", "document_type": "incident_report"},
        )
        retrieved = MiddleAgentOutput(
            query_id=phase1_output.query_id,
            predicted_intent=phase1_output.intent,
            topk_docs=[mock_doc],
            retrieval_diagnostics={"adapter_mode": "faiss_retrieval", "embedding_backend": "tfidf_svd_faiss_fallback"},
        )
        mock_adapter = MagicMock()
        mock_adapter.resolve_output.return_value = MiddleAgentOutput(
            query_id=phase1_output.query_id,
            predicted_intent=phase1_output.intent,
            topk_docs=[],
            retrieval_diagnostics={"adapter_mode": "generated_mock"},
        )
        mock_retrieval = MagicMock()
        mock_retrieval.available = True
        mock_retrieval.retrieve.return_value = retrieved

        output = resolve_phase2_output(
            Settings(),
            phase1_output,
            phase2_adapter=mock_adapter,
            phase2_retrieval=mock_retrieval,
        )
        self.assertEqual(output.topk_docs[0].doc_id, "747383")
        self.assertEqual(output.retrieval_diagnostics.get("adapter_mode"), "faiss_retrieval")
        mock_retrieval.retrieve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
