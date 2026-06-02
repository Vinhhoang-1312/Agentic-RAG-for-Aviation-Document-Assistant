from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aviation_rag.config import Settings
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting


class InputAgentModeTests(unittest.TestCase):
    def test_heuristic_mode_without_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            settings = Settings(
                data_path=temp_path / "missing.csv",
                input_intent_mode="heuristic",
                langsmith_tracing="false",
            )
            phase1 = Phase1HoangIntentRouting(settings)
            output = phase1.build_output("crosswind turbulence final approach")
            self.assertEqual(output.intent, "Metadata_Query")
            self.assertEqual(output.intent_source, "heuristic")

    def test_ml_mode_uses_local_dataset_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "mini_phase1.csv"
            frame = pd.DataFrame(
                [
                    {
                        "report_summary": "maintenance checklist completed after engine warning",
                        "report1_narrative": "engine procedure run by crew",
                        "report2_narrative": "",
                        "primary_problem": "procedure",
                        "event_anomaly": "engine warning",
                        "component_name": "engine",
                        "component_problem": "warning light",
                        "weather_conditions": "",
                        "flight_conditions": "",
                    },
                    {
                        "report_summary": "weather and crosswind on approach",
                        "report1_narrative": "turbulence and runway condition report",
                        "report2_narrative": "",
                        "primary_problem": "weather",
                        "event_anomaly": "turbulence",
                        "component_name": "",
                        "component_problem": "",
                        "weather_conditions": "crosswind turbulence",
                        "flight_conditions": "ifr",
                    },
                ]
            )
            frame.to_csv(dataset_path, index=False)

            settings = Settings(
                data_path=dataset_path,
                input_intent_mode="ml",
                langsmith_tracing="false",
            )
            phase1 = Phase1HoangIntentRouting(settings)
            output = phase1.build_output("engine checklist after warning")
            self.assertIn(output.intent_source, {"ml", "heuristic"})
            self.assertTrue(0.0 <= output.intent_confidence <= 1.0)


if __name__ == "__main__":
    unittest.main()
