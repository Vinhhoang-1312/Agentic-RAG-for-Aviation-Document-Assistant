from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aviation_rag.config import Settings
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting
from aviation_rag.phase2_san_contract_adapter import Phase2SanContractAdapter


class Phase2ContractAdapterTests(unittest.TestCase):
    def test_adapter_generates_mock_when_san_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            settings = Settings(
                data_path=temp_path / "missing.csv",
                phase2_output_path=temp_path / "phase2_san_retrieval_output.jsonl",
                phase2_sample_output_path=temp_path / "missing.sample.jsonl",
                input_intent_mode="heuristic",
                langsmith_tracing="false",
            )
            phase1 = Phase1HoangIntentRouting(settings)
            adapter = Phase2SanContractAdapter(settings)
            phase1_output = phase1.build_output("engine warning checklist")
            phase2_output = adapter.resolve_output(phase1_output)
            self.assertEqual(phase2_output.query_id, phase1_output.query_id)
            self.assertGreater(len(phase2_output.topk_docs), 0)
            self.assertEqual(
                phase2_output.retrieval_diagnostics.get("contract_owner"),
                "Quan San",
            )


if __name__ == "__main__":
    unittest.main()
