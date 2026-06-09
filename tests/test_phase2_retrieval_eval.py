from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aviation_rag.config import Settings
from aviation_rag.phase2_retrieval_eval import (
    evaluate_gold_retrieval_row,
    evaluate_retrieval_gold,
    extracted_retrieved_ids,
)


class Phase2RetrievalEvalTests(unittest.TestCase):
    def test_extracted_retrieved_ids(self):
        row = {
            "topk_docs": [
                {"doc_id": "676109", "chunk_id": "676109#0"},
                {"doc_id": "682094", "chunk_id": "682094#0"},
            ]
        }
        self.assertEqual(extracted_retrieved_ids(row), ["676109", "682094"])

    def test_evaluate_gold_retrieval_row_metrics(self):
        gold_row = {
            "query_id": "q_incident_001",
            "query_raw": "engine failure after takeoff",
            "relevant_doc_ids": ["676109", "682094"],
            "k": 5,
            "minimum_recall_at_k": 0.5,
        }
        metrics = evaluate_gold_retrieval_row(
            ["999999", "676109", "111111", "682094", "222222"],
            gold_row,
            latency_ms=12.5,
            strategy="semantic",
        )
        self.assertEqual(metrics["precision_at_k"], 0.4)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertTrue(metrics["passed"])

    def test_evaluate_retrieval_gold_by_query_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gold_path = Path(temp_dir) / "phase2_gold.jsonl"
            gold_path.write_text(
                '{"query_id":"q_incident_001","query_raw":"engine failure after takeoff with emergency return",'
                '"relevant_doc_ids":["676109","682094"],"k":3,"minimum_recall_at_k":0.33}\n',
                encoding="utf-8",
            )
            report = evaluate_retrieval_gold(
                [
                    {
                        "query_raw": "engine failure after takeoff with emergency return",
                        "topk_docs": [{"doc_id": "676109"}, {"doc_id": "000000"}],
                        "retrieval_diagnostics": {"latency_ms": 5.0, "strategy_requested": "semantic"},
                    }
                ],
                gold_path,
            )
            self.assertEqual(report["gold_rows"], 1)
            self.assertEqual(report["summary"]["mrr"], 1.0)
            self.assertTrue(report["gold_eval_rows"][0]["passed"])

    def test_run_retrieval_gold_benchmark_smoke(self):
        settings = replace(
            Settings(),
            phase1_retrain=False,
            langsmith_tracing="false",
            phase2_embedding_model="tfidf_svd_fallback",
            phase2_index_dir=Settings().artifacts_dir / "phase2_index_fast",
            retrieval_max_docs=2000,
        )
        from aviation_rag.phase2_retrieval_eval import run_retrieval_gold_benchmark

        report = run_retrieval_gold_benchmark(settings, strategies=["bm25", "semantic"])
        self.assertGreaterEqual(report["gold_rows"], 1)
        self.assertIn("bm25", report["strategies"])
        self.assertIn("semantic", report["strategies"])


if __name__ == "__main__":
    unittest.main()
