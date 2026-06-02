from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from aviation_rag.api import create_app
from aviation_rag.config import Settings


class ApiSmokeTests(unittest.TestCase):
    def test_chat_endpoint_runs_full_pipeline(self):
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

            app = create_app(settings=settings)
            client = TestClient(app)

            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json().get("status"), "ok")

            payload = {
                "query": "engine warning checklist",
                "strategy": "bm25",
                "top_k": 5,
                "allow_local_fallback": True,
            }
            chat = client.post("/v1/chat", json=payload)
            self.assertEqual(chat.status_code, 200)
            body = chat.json()
            self.assertIn("query_id", body)
            self.assertIn("answer", body)
            self.assertIn("retrieval_diagnostics", body)
            self.assertTrue((artifacts_dir / "phase3_hoang_grounded_answer_output.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
