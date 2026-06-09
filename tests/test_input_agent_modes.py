from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aviation_rag.config import Settings
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting


class InputAgentModeTests(unittest.TestCase):
    def test_auto_mode_routes_metadata_query(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            settings = Settings(
                data_path=temp_path / "missing.csv",
                input_intent_mode="auto",
                phase1_retrain=True,
                phase1_model_dir=temp_path / "phase1_model",
                langsmith_tracing="false",
            )
            phase1 = Phase1HoangIntentRouting(settings)
            output = phase1.build_output("crosswind turbulence final approach")
            self.assertEqual(output.intent, "Metadata_Query")
            self.assertIn(output.intent_source, {"ml", "heuristic"})

    def test_auto_mode_falls_back_to_heuristic_for_checklist_query(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            settings = Settings(
                data_path=temp_path / "missing.csv",
                input_intent_mode="auto",
                phase1_retrain=True,
                phase1_model_dir=temp_path / "phase1_model",
                langsmith_tracing="false",
            )
            phase1 = Phase1HoangIntentRouting(settings)
            output = phase1.build_output("engine warning checklist after takeoff")
            self.assertEqual(output.intent, "Technical_Procedure")
            self.assertIn(output.intent_source, {"ml", "heuristic"})

    def test_heuristic_mode_does_not_require_ml_agreement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            settings = Settings(
                data_path=temp_path / "missing.csv",
                input_intent_mode="heuristic",
                phase1_retrain=True,
                phase1_model_dir=temp_path / "phase1_model",
                langsmith_tracing="false",
            )
            phase1 = Phase1HoangIntentRouting(settings)
            output = phase1.build_output("what is the meaning of MEL in aviation")
            self.assertEqual(output.intent, "Factoid")
            self.assertEqual(output.intent_source, "heuristic")

    def test_query_only_training_corpus_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            settings = Settings(
                data_path=temp_path / "missing.csv",
                input_intent_mode="ml",
                phase1_retrain=True,
                phase1_model_dir=temp_path / "phase1_model",
                phase1_training_queries_path=temp_path / "training_queries.jsonl",
                langsmith_tracing="false",
            )
            phase1 = Phase1HoangIntentRouting(settings)
            corpus_type = (phase1.intent_model.training_report or {}).get("training_corpus", {}).get("type")
            self.assertEqual(corpus_type, "query_only")
            self.assertLess(phase1.intent_model.training_rows, 500)


if __name__ == "__main__":
    unittest.main()
