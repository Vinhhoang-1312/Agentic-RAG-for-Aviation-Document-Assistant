from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aviation_rag.config import Settings
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting
from aviation_rag.phase2_san_faiss_retrieval import Phase2SanFaissRetrieval

HAS_FAISS = importlib.util.find_spec("faiss") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(HAS_FAISS and HAS_SKLEARN, "Phase 2 retrieval dependencies are not installed.")
class Phase2FaissRetrievalTests(unittest.TestCase):
    def _settings_with_dataset(self, temp_path: Path) -> Settings:
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
                    "event_anomaly": "engine warning checklist",
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
                {
                    "event_id": "event_incident",
                    "primary_problem": "aircraft",
                    "weather_conditions": "",
                    "flight_conditions": "vfr",
                    "component_name": "engine",
                    "component_problem": "failure",
                    "event_anomaly": "emergency return",
                    "report_summary": "engine failure after takeoff led to emergency return",
                    "report1_narrative": "pilot declared emergency and returned to departure airport",
                    "report2_narrative": "",
                    "location_airport": "DAD",
                    "location_state": "",
                },
            ]
        )
        frame.to_csv(dataset_path, index=False)
        return Settings(
            artifacts_dir=temp_path / "artifacts",
            data_path=dataset_path,
            phase2_index_dir=temp_path / "phase2_index",
            phase2_output_path=temp_path / "phase2.jsonl",
            input_intent_mode="heuristic",
            langsmith_tracing="false",
            retrieval_max_docs=100,
            retrieval_tfidf_max_features=512,
            retrieval_svd_components=8,
            phase2_embedding_model="tfidf_svd_fallback",
        )

    def test_retrieval_returns_real_docs_from_local_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = self._settings_with_dataset(Path(temp_dir))
            phase1 = Phase1HoangIntentRouting(settings)
            retrieval = Phase2SanFaissRetrieval(settings)

            phase1_output = phase1.build_output("engine warning checklist", strategy="bm25")
            phase2_output = retrieval.retrieve(phase1_output)

            self.assertEqual(phase2_output.retrieval_diagnostics.get("adapter_mode"), "faiss_retrieval")
            self.assertIn("embedding_backend", phase2_output.retrieval_diagnostics)
            self.assertGreater(len(phase2_output.topk_docs), 0)
            self.assertEqual(phase2_output.topk_docs[0].doc_id, "event_engine")

    def test_metadata_query_applies_document_type_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = self._settings_with_dataset(Path(temp_dir))
            phase1 = Phase1HoangIntentRouting(settings)
            retrieval = Phase2SanFaissRetrieval(settings)

            phase1_output = phase1.build_output("crosswind turbulence runway weather", strategy="metadata_first")
            phase2_output = retrieval.retrieve(phase1_output)

            self.assertTrue(phase2_output.retrieval_diagnostics.get("metadata_filter_applied"))
            self.assertEqual(phase2_output.topk_docs[0].metadata.get("document_type"), "metadata")

    def test_hybrid_rrf_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = self._settings_with_dataset(Path(temp_dir))
            phase1 = Phase1HoangIntentRouting(settings)
            retrieval = Phase2SanFaissRetrieval(settings)

            phase1_output = phase1.build_output("engine failure after takeoff emergency return", strategy="hybrid_rrf")
            first = retrieval.retrieve(phase1_output)
            second = retrieval.retrieve(phase1_output)

            self.assertEqual(
                [doc.doc_id for doc in first.topk_docs],
                [doc.doc_id for doc in second.topk_docs],
            )
            self.assertEqual(first.retrieval_diagnostics.get("fusion_method"), "reciprocal_rank_fusion")

    def test_diagnostics_include_streamlit_method_proof_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = self._settings_with_dataset(Path(temp_dir))
            phase1 = Phase1HoangIntentRouting(settings)
            retrieval = Phase2SanFaissRetrieval(settings)
            output = retrieval.retrieve(phase1.build_output("engine checklist", strategy="hybrid"))
            diagnostics = output.retrieval_diagnostics
            for key in [
                "retrieval_backend",
                "embedding_model",
                "embedding_dim",
                "faiss_index_type",
                "normalization",
                "chunk_count",
                "bm25_enabled",
                "metadata_filter_applied",
                "fusion_method",
                "latency_ms",
            ]:
                self.assertIn(key, diagnostics)


if __name__ == "__main__":
    unittest.main()
