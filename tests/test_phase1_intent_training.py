from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aviation_rag.config import Settings
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting
from aviation_rag.phase1_intent_training import (
    _normalized_query_key,
    build_training_corpus,
    evaluate_gold_labels,
    evaluate_heuristic_gold_labels,
    load_intent_model,
    preprocess_for_intent_ml,
    train_intent_model,
)
from aviation_rag.phase3_grounding_eval import evaluate_grounding_gold


class Phase1IntentTrainingTests(unittest.TestCase):
    def test_preprocess_applies_stemming(self):
        raw = "running engines procedures"
        stemmed = preprocess_for_intent_ml(raw, use_stemming=True)
        self.assertNotEqual(stemmed, raw)
        self.assertIn("engin", stemmed)

    def test_build_training_corpus_is_query_only_and_holds_out_gold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gold_path = temp_path / "gold.jsonl"
            gold_path.write_text(
                '{"query_id":"g1","query_raw":"engine checklist after warning","expected_intent":"Technical_Procedure"}\n',
                encoding="utf-8",
            )
            training_path = temp_path / "training_queries.jsonl"
            training_path.write_text(
                '{"query_raw":"engine checklist after warning","intent":"Technical_Procedure"}\n'
                '{"query_raw":"weather turbulence on final","intent":"Metadata_Query"}\n',
                encoding="utf-8",
            )
            settings = Settings(
                phase1_gold_labels_path=gold_path,
                phase1_training_queries_path=training_path,
                langsmith_tracing="false",
            )
            texts, labels, metadata = build_training_corpus(settings)
            self.assertEqual(metadata["type"], "query_only")
            self.assertGreaterEqual(len(texts), 20)
            holdout_query = "engine checklist after warning"
            self.assertTrue(
                all(_normalized_query_key(text) != _normalized_query_key(holdout_query) for text in texts)
            )

    def test_train_validation_split_and_persist_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gold_path = temp_path / "gold.jsonl"
            gold_path.write_text(
                '{"query_id":"g1","query_raw":"engine checklist after warning","expected_intent":"Technical_Procedure"}\n',
                encoding="utf-8",
            )
            training_path = temp_path / "training_queries.jsonl"
            training_path.write_text(
                "\n".join(
                    [
                        '{"query_raw":"hydraulic leak troubleshooting procedure","intent":"Technical_Procedure"}',
                        '{"query_raw":"runway contamination during landing","intent":"Metadata_Query"}',
                        '{"query_raw":"bird strike rejected takeoff","intent":"Incident_Report"}',
                        '{"query_raw":"what does tcas mean","intent":"Factoid"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            settings = Settings(
                data_path=temp_path / "missing.csv",
                phase1_model_dir=temp_path / "model",
                phase1_gold_labels_path=gold_path,
                phase1_training_queries_path=training_path,
                phase1_validation_split=0.25,
                phase1_retrain=True,
                phase1_use_stemming=True,
                langsmith_tracing="false",
            )
            model = train_intent_model(settings)
            self.assertGreater(model.training_rows, 0)
            validation = (model.training_report or {}).get("validation_metrics", {})
            self.assertIn("validation_accuracy", validation)
            self.assertEqual((model.training_report or {}).get("training_corpus", {}).get("type"), "query_only")
            self.assertTrue((settings.phase1_model_dir / "tfidf_vectorizer.joblib").exists())
            self.assertTrue((settings.phase1_model_dir / "logistic_classifier.joblib").exists())
            self.assertTrue((settings.phase1_model_dir / "preprocessing_pipeline.joblib").exists())
            self.assertTrue((settings.phase1_model_dir / "training_report.json").exists())

            cached = load_intent_model(settings.phase1_model_dir)
            self.assertIsNotNone(cached)
            gold = evaluate_gold_labels(cached, gold_path, use_stemming=True)
            self.assertEqual(gold["gold_rows"], 1)
            heuristic_gold = evaluate_heuristic_gold_labels(gold_path)
            self.assertEqual(heuristic_gold["gold_rows"], 1)

    def test_auto_mode_uses_gold_procedure_query(self):
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


class Phase3GroundingEvalTests(unittest.TestCase):
    def test_evaluate_grounding_gold_by_query_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gold_path = Path(temp_dir) / "phase3_gold.jsonl"
            gold_path.write_text(
                '{"query_id":"gold_x","query_raw":"engine failure after takeoff","min_citations":1,"max_hallucination_risk":0.9,"require_non_empty_answer":true}\n',
                encoding="utf-8",
            )
            rows = [
                {
                    "query_id": "q_other",
                    "query_raw": "engine failure after takeoff",
                    "answer": "engine failure after takeoff with emergency return",
                    "citations": [{"doc_id": "1", "reason": "evidence"}],
                    "hallucination_risk": 0.4,
                }
            ]
            report = evaluate_grounding_gold(rows, gold_path)
            self.assertEqual(report["gold_rows"], 1)
            self.assertEqual(report["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
