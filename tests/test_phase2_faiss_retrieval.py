from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aviation_rag.config import Settings
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting
from aviation_rag.phase2_san_faiss_retrieval import Phase2SanFaissRetrieval


class Phase2FaissRetrievalTests(unittest.TestCase):
    def test_retrieval_returns_real_docs_from_local_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "mini_retrieval.csv"
            frame = pd.DataFrame(
                [
                    {
                        "event_id": "event_engine",
                        "primary_problem": "procedure",
                        "weather_conditions": "",
                        "flight_conditions": "",
                        "component_name": "engine",
                        "component_problem": "warning light",
                        "event_anomaly": "engine warning",
                        "report_summary": "engine warning checklist used by crew",
                        "report1_narrative": "crew followed checklist after engine oil pressure warning",
                        "report2_narrative": "",
                        "location_airport": "SGN",
                        "location_state": "",
                    },
                    {
                        "event_id": "event_weather",
                        "primary_problem": "weather",
                        "weather_conditions": "crosswind turbulence",
                        "flight_conditions": "ifr",
                        "component_name": "",
                        "component_problem": "",
                        "event_anomaly": "turbulence",
                        "report_summary": "crosswind turbulence during final approach",
                        "report1_narrative": "weather conditions changed near runway",
                        "report2_narrative": "",
                        "location_airport": "HAN",
                        "location_state": "",
                    },
                ]
            )
            frame.to_csv(dataset_path, index=False)

            settings = Settings(
                data_path=dataset_path,
                input_intent_mode="heuristic",
                langsmith_tracing="false",
                retrieval_max_docs=100,
                retrieval_tfidf_max_features=512,
                retrieval_svd_components=8,
            )
            phase1 = Phase1HoangIntentRouting(settings)
            retrieval = Phase2SanFaissRetrieval(settings)

            phase1_output = phase1.build_output("engine warning checklist")
            phase2_output = retrieval.retrieve(phase1_output)

            self.assertEqual(phase2_output.retrieval_diagnostics.get("adapter_mode"), "faiss_retrieval")
            self.assertGreater(len(phase2_output.topk_docs), 0)
            self.assertEqual(phase2_output.topk_docs[0].doc_id, "event_engine")


if __name__ == "__main__":
    unittest.main()
