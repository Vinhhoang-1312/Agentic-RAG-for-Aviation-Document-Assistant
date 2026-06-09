from __future__ import annotations

import unittest
from pathlib import Path

from aviation_rag.config import Settings
from aviation_rag.gold_ui import gold_grounding_match, gold_intent_match, gold_retrieval_match


class GoldUiTests(unittest.TestCase):
    def test_gold_intent_match_for_preset_query(self):
        settings = Settings()
        result = {
            "query_raw": "engine failure after takeoff with emergency return",
            "intent": "Incident_Report",
            "retrieval_plan": {"strategy": "semantic"},
        }
        match = gold_intent_match(result, settings.phase1_gold_labels_path)
        self.assertIsNotNone(match)
        self.assertTrue(match["correct"])
        self.assertEqual(match["expected_intent"], "Incident_Report")

    def test_gold_intent_match_returns_none_for_unknown_query(self):
        settings = Settings()
        match = gold_intent_match({"query_raw": "totally unknown query xyz"}, settings.phase1_gold_labels_path)
        self.assertIsNone(match)

    def test_gold_grounding_match_uses_current_query_not_first_gold_row(self):
        settings = Settings()
        result = {
            "query_raw": "what is the meaning of MEL in aviation?",
            "query_id": "q_test",
            "answer": "MEL is Minimum Equipment List.",
            "citations": [{"doc_id": "1", "reason": "evidence"}],
            "hallucination_risk": 0.2,
        }
        match = gold_grounding_match(result, settings.phase3_gold_labels_path)
        self.assertIsNotNone(match)
        self.assertEqual(match["query_raw"], result["query_raw"])
        self.assertTrue(match["passed"])

    def test_gold_grounding_match_returns_none_for_unknown_query(self):
        settings = Settings()
        match = gold_grounding_match(
            {"query_raw": "unknown", "answer": "x", "citations": [], "hallucination_risk": 0.5},
            settings.phase3_gold_labels_path,
        )
        self.assertIsNone(match)

    def test_gold_retrieval_match_for_preset_query(self):
        settings = Settings()
        result = {
            "query_raw": "engine failure after takeoff with emergency return",
            "topk_docs": [{"doc_id": "676109"}, {"doc_id": "000000"}],
            "retrieval_plan": {"strategy": "semantic"},
            "retrieval_diagnostics": {"latency_ms": 10.0},
        }
        match = gold_retrieval_match(result, settings.phase2_gold_labels_path)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match["recall_at_k"], 0.0)


if __name__ == "__main__":
    unittest.main()
