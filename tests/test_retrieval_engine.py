"""
Unit tests for Phase 2 Retrieval Engine (Quan San).
Tests the engine, adapter fallback, and contract compliance.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aviation_rag.config import Settings
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting
from aviation_rag.phase2_san_contract_adapter import Phase2SanContractAdapter
from aviation_rag.schemas import MiddleAgentOutput, RetrievedDoc


class TestAdapterFallback(unittest.TestCase):
    """Test that the adapter falls back gracefully when index is missing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.settings = Settings(
            data_path=Path(self.temp_dir) / "missing.csv",
            phase2_output_path=Path(self.temp_dir) / "phase2_output.jsonl",
            phase2_sample_output_path=Path(self.temp_dir) / "missing.sample.jsonl",
            index_dir=Path(self.temp_dir) / "index_store",
            input_intent_mode="heuristic",
            langsmith_tracing="false",
        )

    def test_mock_fallback_produces_valid_output(self):
        """When index doesn't exist, adapter should return mock data."""
        phase1 = Phase1HoangIntentRouting(self.settings)
        adapter = Phase2SanContractAdapter(self.settings)

        phase1_output = phase1.build_output("engine failure after takeoff")
        result = adapter.resolve_output(phase1_output)

        self.assertIsInstance(result, MiddleAgentOutput)
        self.assertEqual(result.query_id, phase1_output.query_id)
        self.assertGreater(len(result.topk_docs), 0)
        self.assertEqual(result.retrieval_diagnostics["contract_owner"], "Quan San")

    def test_mock_covers_all_intents(self):
        """Mock fallback should work for all 4 intent types."""
        phase1 = Phase1HoangIntentRouting(self.settings)
        adapter = Phase2SanContractAdapter(self.settings)

        queries = {
            "Incident_Report": "engine failure after takeoff",
            "Technical_Procedure": "engine warning checklist procedure",
            "Metadata_Query": "weather conditions at airport",
            "Factoid": "what is the ICAO code for LAX",
        }

        for expected_intent, query in queries.items():
            with self.subTest(intent=expected_intent):
                phase1_output = phase1.build_output(query)
                result = adapter.resolve_output(phase1_output)

                self.assertIsInstance(result, MiddleAgentOutput)
                self.assertGreater(len(result.topk_docs), 0)

                # Verify doc contract
                for doc in result.topk_docs:
                    self.assertIsInstance(doc, RetrievedDoc)
                    self.assertTrue(doc.doc_id)
                    self.assertTrue(doc.chunk_id)
                    self.assertTrue(doc.chunk_text)
                    self.assertIn("final", doc.scores)

    def test_adapter_mode_field_present(self):
        """Diagnostics should always include adapter_mode."""
        phase1 = Phase1HoangIntentRouting(self.settings)
        adapter = Phase2SanContractAdapter(self.settings)

        result = adapter.resolve_output(phase1.build_output("bird strike on approach"))
        self.assertIn("adapter_mode", result.retrieval_diagnostics)


class TestRetrievedDocContract(unittest.TestCase):
    """Test that RetrievedDoc schema matches the shared contract."""

    def test_doc_schema_validation(self):
        doc = RetrievedDoc(
            doc_id="asrs_12345",
            chunk_id="asrs_12345#0",
            chunk_text="Test chunk text about engine failure.",
            scores={"semantic": 0.85, "bm25": 0.30, "final": 0.72},
            metadata={"source": "phase2_real_retrieval", "event_id": "12345"},
        )
        self.assertEqual(doc.doc_id, "asrs_12345")
        self.assertEqual(doc.scores["final"], 0.72)

    def test_doc_serialization(self):
        doc = RetrievedDoc(
            doc_id="test_001",
            chunk_id="test_001#0",
            chunk_text="Test text.",
            scores={"semantic": 0.9, "final": 0.9},
            metadata={},
        )
        d = doc.model_dump()
        self.assertIn("doc_id", d)
        self.assertIn("scores", d)
        # Round-trip
        doc2 = RetrievedDoc.model_validate(d)
        self.assertEqual(doc2.doc_id, doc.doc_id)


if __name__ == "__main__":
    unittest.main()
