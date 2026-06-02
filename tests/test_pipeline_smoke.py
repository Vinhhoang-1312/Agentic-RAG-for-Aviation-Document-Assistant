from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aviation_rag.config import Settings

try:
    from aviation_rag.graph import build_graph

    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False


@unittest.skipUnless(HAS_LANGGRAPH, "LangGraph is not installed in current environment.")
class PipelineSmokeTests(unittest.TestCase):
    def test_end_to_end_with_phase2_mock_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            artifacts_dir = temp_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            settings = Settings(
                artifacts_dir=artifacts_dir,
                data_path=temp_path / "missing.csv",
                phase1_output_path=artifacts_dir / "phase1_hoang_intent_routing_output.jsonl",
                phase2_output_path=artifacts_dir / "phase2_san_retrieval_output.jsonl",
                phase2_sample_output_path=artifacts_dir / "phase2_san_retrieval_output.sample.jsonl",
                phase3_output_path=artifacts_dir / "phase3_hoang_grounded_answer_output.jsonl",
                openai_api_key=None,
                langsmith_tracing="false",
                input_intent_mode="heuristic",
            )
            graph = build_graph(settings)
            state = {
                "query_raw": "engine warning checklist",
                "write_phase1_artifact": True,
                "write_phase2_artifact": True,
                "write_phase3_artifact": True,
                "allow_local_fallback": True,
            }
            result = graph.invoke(state)
            self.assertIn("answer", result)
            self.assertIn("hallucination_risk", result)
            self.assertTrue((artifacts_dir / "phase1_hoang_intent_routing_output.jsonl").exists())
            self.assertTrue((artifacts_dir / "phase2_san_retrieval_output.jsonl").exists())
            self.assertTrue((artifacts_dir / "phase3_hoang_grounded_answer_output.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
