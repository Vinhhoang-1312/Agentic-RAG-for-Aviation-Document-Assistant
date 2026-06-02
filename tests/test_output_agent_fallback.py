from __future__ import annotations

import unittest
from unittest.mock import patch

from aviation_rag.config import Settings
from aviation_rag.phase3_hoang_grounded_qa import Phase3HoangGroundedQA
from aviation_rag.schemas import MiddleAgentOutput, RetrievedDoc


class OutputAgentFallbackTests(unittest.TestCase):
    def test_fallback_when_openai_call_fails(self):
        settings = Settings(openai_api_key="fake-key")
        phase3 = Phase3HoangGroundedQA(settings)
        phase2_output = MiddleAgentOutput(
            query_id="q_test_1",
            predicted_intent="Technical_Procedure",
            topk_docs=[
                RetrievedDoc(
                    doc_id="doc_E1",
                    chunk_id="doc_E1#0",
                    chunk_text="Engine warning checklist and maintenance procedure text.",
                    scores={"bm25": 0.6, "semantic": 0.5, "final": 0.56},
                    metadata={},
                )
            ],
            retrieval_diagnostics={},
        )

        with patch.object(phase3, "_call_openai", side_effect=RuntimeError("network down")):
            output = phase3.generate(
                question="engine warning checklist",
                middle_output=phase2_output,
                allow_fallback=True,
            )

        self.assertEqual(output.query_id, "q_test_1")
        self.assertTrue(len(output.answer) > 0)
        self.assertTrue(len(output.citations) > 0)


if __name__ == "__main__":
    unittest.main()
