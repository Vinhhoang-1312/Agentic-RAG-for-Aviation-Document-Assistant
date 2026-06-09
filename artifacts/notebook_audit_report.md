# Notebook Audit Report

## phase1_hoang_intent_routing_research.ipynb

- Cell 00 `markdown`: `skipped` (0.0 ms)
- Cell 01 `code`: `ok` (23827.0 ms)

```text
Project root: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project
Data path: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\data\kaggle\ASRS-clean-dataset-aviation-safety.csv
Phase 1 artifact: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase1_hoang_intent_routing_output.jsonl
Saved model dir: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase1_intent_model
Gold labels: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\data\phase1_intent_gold_labels.jsonl
Intent architecture: TF-IDF + Logistic Regression
Intent runtime mode: ml
Training mode: tfidf_logistic_regression
Training rows: 5029
Training label counts: {'Incident_Report': 2186, 'Technical_Procedure': 1457, 'Metadata_Query': 1380, 'Factoid': 6}
Preprocessing: {'normalize_text': True, 'aviation_jargon_expansion': True, 'english_stemming': True, 'stemmer_backend': 'nltk_snowball_english'}
Validation accuracy: 0.6252
Validation macro F1: 0.4633
Gold-set accuracy: 0.875 (8 queries)
Summary: {'training_rows': 5029, 'label_counts': {'Incident_Report': 
```

- Cell 02 `markdown`: `skipped` (0.0 ms)
- Cell 03 `code`: `ok` (21.6 ms)

```text
Saved pipeline files:
 - C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase1_intent_model\preprocessing_pipeline.joblib
 - C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase1_intent_model\tfidf_vectorizer.joblib
 - C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase1_intent_model\logistic_classifier.joblib
 - C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase1_intent_model\training_report.json
                metric      value
0  validation_accuracy     0.6252
1  validation_macro_f1     0.4633
2        gold_accuracy     0.8750
3           train_rows  4023.0000
4      validation_rows  1006.0000
          query_id      expected_intent  ... confidence  correct
0   q_incident_001      Incident_Report  ...     0.5491     True
1  q_procedure_001  Technical_Procedure  ...     0.5313     True
2   q_metadata_001       Metadata_Query  ...     0.4883     True
3    q_factoid_001              Factoid  ...     0.4793     True
4   q_incident_002      Incident_Report  ...     0.5299     True
5  q_procedure_002  Technical_
```

- Cell 04 `markdown`: `skipped` (0.0 ms)
- Cell 05 `markdown`: `skipped` (0.0 ms)
- Cell 06 `code`: `ok` (5.2 ms)

```text
query_id  ... expected_strategy
0   q_incident_001  ...          semantic
1  q_procedure_001  ...              bm25
2   q_metadata_001  ...    metadata_first
3    q_factoid_001  ...          semantic

[4 rows x 4 columns]
```

- Cell 07 `markdown`: `skipped` (0.0 ms)
- Cell 08 `code`: `ok` (21869.2 ms)

```text
method  ...     confidence_source
0  TF-IDF + Logistic Regression  ...     model probability
1            Heuristic baseline  ...  fixed baseline value

[2 rows x 4 columns]
          query_id  ...                                    rewritten_query
0   q_incident_001  ...  aviation incident narrative lookup for: engine...
1  q_procedure_001  ...  aviation troubleshooting and procedure lookup ...
2   q_metadata_001  ...  aviation metadata and operating condition look...
3    q_factoid_001  ...  direct aviation fact lookup for: what is the m...

[4 rows x 11 columns]
```

- Cell 09 `markdown`: `skipped` (0.0 ms)
- Cell 10 `code`: `ok` (9.3 ms)

```text
query_id  ...                                     routing_reason
0   q_incident_001  ...  Narrative incident queries benefit from semant...
1  q_procedure_001  ...  Procedure-style queries favor keyword-heavy ch...
2   q_metadata_001  ...  Metadata queries should prioritize structured ...
3    q_factoid_001  ...  Factoid queries need concise semantic lookup w...

[4 rows x 6 columns]
```

- Cell 11 `markdown`: `skipped` (0.0 ms)
- Cell 12 `code`: `ok` (29252.4 ms)

```text
=== Tổng hợp BM25 vs Semantic ===
                                   query  ...  semantic_weight_dominant_score
0  qrh checklist for engine fire warning  ...                             1.0

[1 rows x 8 columns]
=== Top-1 theo strategy ===
   strategy  uses_bm25_in_final  ... metadata_score final_score
0      bm25                True  ...           0.75      0.8211
1  semantic               False  ...           0.40      0.9300

[2 rows x 10 columns]
=== Top-3 chi tiết ===
```

- Cell 13 `markdown`: `skipped` (0.0 ms)
- Cell 14 `code`: `ok` (2.1 ms)

```text
Wrote 4 rows to C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase1_hoang_intent_routing_output.jsonl
Phase 1 contract validation passed.
None
```

- Cell 15 `markdown`: `skipped` (0.0 ms)
- Cell 16 `code`: `ok` (11.5 ms)

```text
{'sample_size': 4, 'intent_accuracy': np.float64(1.0), 'routing_accuracy': np.float64(1.0), 'confidence_min': np.float64(0.4793), 'confidence_mean': np.float64(0.512), 'confidence_max': np.float64(0.5491)}
          query_id  ... confidence
0   q_incident_001  ...     0.5491
1  q_procedure_001  ...     0.5313
2   q_metadata_001  ...     0.4883
3    q_factoid_001  ...     0.4793

[4 rows x 9 columns]
```


## phase2_san_semantic_retrieval_research.ipynb

- Cell 00 `markdown`: `skipped` (0.0 ms)
- Cell 01 `markdown`: `skipped` (0.0 ms)
- Cell 02 `code`: `ok` (5.0 ms)

```text
{'data_path': 'C:\\Users\\DELL\\Desktop\\Vinh Hoang\\Master Program\\Xử lý ngôn ngữ tự nhiên\\Project\\data\\kaggle\\ASRS-clean-dataset-aviation-safety.csv', 'phase2_index_dir': 'C:\\Users\\DELL\\Desktop\\Vinh Hoang\\Master Program\\Xử lý ngôn ngữ tự nhiên\\Project\\artifacts\\phase2_index', 'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2', 'retrieval_max_docs': 15000, 'langsmith_tracing': 'true', 'langsmith_api_key': 'configured'}
```

- Cell 03 `markdown`: `skipped` (0.0 ms)
- Cell 04 `code`: `ok` (64577.3 ms)

```text
chunk_count: 16713
first_chunk:
{
  "doc_id": "645583",
  "chunk_id": "645583#0",
  "chunk_text": "a b757-200 was dispatched with deferred non airworthy fuselage skin scratches near the l static port in the rvsm critical area. soft straight hair lines on acft skin below l static port area suspected to be caused by other station's lavatory svcing truck safety fiberglas flag poles contacting acft when apching forward lavatory svcing panel (as per auditor). formal request for burnishing non airworthy of flt marks on fuselage skin not documented on non routine card of technician's non routine write-up form. foreman deferral burnishing carried out by following station zzz1. gpm section 17-03 has been reviewed for proper line station maint item deferment procs.",
  "lexical_text": "a b757 200 was dispatched with deferred non airworthy fuselage skin scratches near the l static port in the rvsm critical area soft straight hair lines on aircraft skin below l static port area suspected to be caused by other station s lavatory svcing truck safety fiberglas flag poles contacting aircraft when apching forward lavatory svcing panel as per auditor formal request for burnishing non airworthy of fl
```

- Cell 05 `markdown`: `skipped` (0.0 ms)
- Cell 06 `code`: `ok` (519.7 ms)

```text
{
  "query_id": "q_894192b0",
  "query_raw": "engine warning checklist after takeoff",
  "query_normalized": "engine warning checklist after takeoff",
  "intent": "Incident_Report",
  "intent_confidence": 0.42972178485072937,
  "intent_source": "ml",
  "expanded_queries": [
    "engine warning checklist after takeoff",
    "engine warning checklist after takeoff aviation incident report",
    "engine warning checklist after takeoff event narrative",
    "engine warning checklist after takeoff safety occurrence summary"
  ],
  "rewritten_query": "aviation incident narrative lookup for: engine warning checklist after takeoff",
  "retrieval_plan": {
    "strategy": "semantic",
    "fallback_strategy": "hybrid",
    "top_k": 5,
    "filters": {},
    "routing_reason": "Narrative incident queries benefit from semantic similarity over safety reports."
  },
  "created_at": "2026-06-06T06:16:02.542312"
}
None
```

- Cell 07 `markdown`: `skipped` (0.0 ms)
- Cell 08 `code`: `ok` (33298.2 ms)

```text
available: True
build_error: None
index_info:
{
  "retrieval_backend": "phase2_dense_bm25_hybrid",
  "embedding_model": "fallback:sentence-transformers/all-MiniLM-L6-v2",
  "embedding_backend": "tfidf_svd_faiss_fallback",
  "embedding_dim": 128,
  "faiss_index_type": "IndexFlatIP",
  "normalization": "L2",
  "chunk_count": 16713,
  "bm25_enabled": true,
  "index_dir": "C:\\Users\\DELL\\Desktop\\Vinh Hoang\\Master Program\\Xử lý ngôn ngữ tự nhiên\\Project\\artifacts\\phase2_index",
  "fallback_reason": "sentence-transformers is not installed."
}
None
```

- Cell 09 `markdown`: `skipped` (0.0 ms)
- Cell 10 `code`: `ok` (281.8 ms)

```text
diagnostics:
{
  "adapter_mode": "faiss_retrieval",
  "contract_owner": "Quang San",
  "retrieval_backend": "phase2_dense_bm25_hybrid",
  "embedding_model": "fallback:sentence-transformers/all-MiniLM-L6-v2",
  "embedding_backend": "tfidf_svd_faiss_fallback",
  "embedding_dim": 128,
  "faiss_index_type": "IndexFlatIP",
  "normalization": "L2",
  "chunk_count": 16713,
  "corpus_size": 16713,
  "bm25_enabled": true,
  "metadata_filter_applied": false,
  "fusion_method": "weighted_linear_fusion",
  "score_weights": {
    "semantic": 0.5,
    "bm25": 0.35,
    "metadata": 0.15
  },
  "strategy_requested": "hybrid",
  "fallback_strategy": "hybrid",
  "routing_reason": "Manual override requested. Original routing intent was Incident_Report.",
  "dataset_path": "C:\\Users\\DELL\\Desktop\\Vinh Hoang\\Master Program\\Xử lý ngôn ngữ tự nhiên\\Project\\data\\kaggle\\ASRS-clean-dataset-aviation-safety.csv",
  "index_dir": "C:\\Users\\DELL\\Desktop\\Vinh Hoang\\Master Program\\Xử lý ngôn ngữ tự nhiên\\Project\\artifacts\\phase2_index",
  "latency_ms": 272.6830999890808,
  "fallback_reason": "sentence-transformers is not installed."
}
top docs:
1 721923 incident_report {'rank': 1.0, 'semantic': 0
```

- Cell 11 `markdown`: `skipped` (0.0 ms)
- Cell 12 `code`: `ok` (1332.9 ms)

```text
[{'strategy': 'bm25', 'top_doc': '721923', 'top_type': 'incident_report', 'top_final': 0.9346283674240112, 'fusion': 'weighted_linear_fusion', 'latency_ms': 280.165100004524, 'embedding_backend': 'tfidf_svd_faiss_fallback'}, {'strategy': 'semantic', 'top_doc': '748920', 'top_type': 'incident_report', 'top_final': 0.8500000238418579, 'fusion': 'weighted_linear_fusion', 'latency_ms': 242.13959998451173, 'embedding_backend': 'tfidf_svd_faiss_fallback'}, {'strategy': 'hybrid', 'top_doc': '721923', 'top_type': 'incident_report', 'top_final': 0.7630017995834351, 'fusion': 'weighted_linear_fusion', 'latency_ms': 231.95210000267252, 'embedding_backend': 'tfidf_svd_faiss_fallback'}, {'strategy': 'metadata_first', 'top_doc': '721923', 'top_type': 'incident_report', 'top_final': 0.8928954601287842, 'fusion': 'weighted_linear_fusion', 'latency_ms': 247.50660001882352, 'embedding_backend': 'tfidf_svd_faiss_fallback'}, {'strategy': 'hybrid_rrf', 'top_doc': '746366', 'top_type': 'incident_report', 'top_final': 1.0, 'fusion': 'reciprocal_rank_fusion', 'latency_ms': 309.43890000344254, 'embedding_backend': 'tfidf_svd_faiss_fallback'}]
```

- Cell 13 `markdown`: `skipped` (0.0 ms)
- Cell 14 `code`: `ok` (897.6 ms)

```text
[{'query': 'engine warning checklist after takeoff', 'strategy': 'bm25', 'precision_at_k': 1.0, 'recall_at_k': 1.0, 'mrr': 1.0, 'latency_ms': 236.4260999893304}, {'query': 'crosswind turbulence during final approach', 'strategy': 'metadata_first', 'precision_at_k': 1.0, 'recall_at_k': 1.0, 'mrr': 1.0, 'latency_ms': 300.72920001111925}, {'query': 'engine failure after takeoff with emergency return', 'strategy': 'semantic', 'precision_at_k': 0.6, 'recall_at_k': 1.0, 'mrr': 1.0, 'latency_ms': 346.4240000175778}]
```

- Cell 15 `markdown`: `skipped` (0.0 ms)
- Cell 16 `code`: `ok` (12680.9 ms)

```text
Gold labels: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\data\phase2_retrieval_gold_labels.jsonl
Saved report: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase2_retrieval_gold_report.json
Overall summary: {'precision_at_k': 0.03, 'recall_at_k': 0.05, 'mrr': 0.0833, 'latency_ms': 313.53}
         strategy  pass_rate  recall_at_k  precision_at_k     mrr  latency_ms
0            bm25      0.375       0.1250           0.075  0.2188    329.9962
1        semantic      0.000       0.0000           0.000  0.0000    303.5725
2          hybrid      0.125       0.0417           0.025  0.0417    306.7425
3  metadata_first      0.000       0.0000           0.000  0.0000    275.1450
4      hybrid_rrf      0.250       0.0833           0.050  0.1562    352.1937
```

- Cell 17 `markdown`: `skipped` (0.0 ms)
- Cell 18 `code`: `ok` (4.0 ms)

```text
wrote: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase2_san_retrieval_output.jsonl
{"query_id": "q_6592b838", "predicted_intent": "Incident_Report", "topk_docs": [{"doc_id": "721923", "chunk_id": "721923#0", "chunk_text": "a b737-700 plt describes departing with the ap 'stab trim' switch inadvertently 'off' preventing rvsm op. bottom line was we missed some switch positions after maint was done on acft. this was the culmination of an extended maint delay and long sit time. although we were not fatigued we were definitely tired. duty start: xd:22. i had awakened that day at xa:00. landed in zzz1 at xf:30. acft change and scheduled to depart at xi:00. our acft arrived with a 'display source' warning on both plts outer display unit. extended gnd time awaiting part. finally pushed back around xn:00. after engine start had 'master caution' and 'eng' annunciator illuminate. noted that both number one and two engine electronic eng controls were in 'altn' mode. should have just selected 'on' for both electronic eng controls but interpreted this as a malfunction. this the result of the previous display electronic unit malfunction and maint wor
```

- Cell 19 `markdown`: `skipped` (0.0 ms)
- Cell 20 `markdown`: `skipped` (0.0 ms)
- Cell 21 `code`: `ok` (839.6 ms)

```text
{
  "summary": {
    "precision_at_k": 0.8667,
    "recall_at_k": 1.0,
    "mrr": 1.0,
    "latency_ms": 275.3725
  },
  "rows": [
    {
      "query": "engine warning checklist after takeoff",
      "strategy": "bm25",
      "top_doc": "721923",
      "term_hits_in_topk": 5,
      "precision_at_5": 1.0,
      "recall_at_5": 1.0,
      "mrr": 1.0,
      "latency_ms": 236.88
    },
    {
      "query": "crosswind turbulence during final approach",
      "strategy": "metadata_first",
      "top_doc": "745245",
      "term_hits_in_topk": 5,
      "precision_at_5": 1.0,
      "recall_at_5": 1.0,
      "mrr": 1.0,
      "latency_ms": 260.02
    },
    {
      "query": "engine failure after takeoff with emergency return",
      "strategy": "semantic",
      "top_doc": "733684",
      "term_hits_in_topk": 3,
      "precision_at_5": 0.6,
      "recall_at_5": 1.0,
      "mrr": 1.0,
      "latency_ms": 329.22
    }
  ]
}
None
```

- Cell 22 `markdown`: `skipped` (0.0 ms)

## phase3_hoang_grounded_output_research.ipynb

- Cell 00 `markdown`: `skipped` (0.0 ms)
- Cell 01 `code`: `ok` (10985.5 ms)

```text
Run mode: Fast local
force_local: True
Embedding: tfidf_svd_fallback
Index: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase2_index_fast
Phase 1 artifact: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase1_hoang_intent_routing_output.jsonl
Phase 2 artifact: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase2_san_retrieval_output.jsonl
Phase 3 artifact: C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase3_hoang_grounded_answer_output.jsonl
Retrieval available: True
Mock library file: aviation_rag/phase2_san_contract_adapter.py
None
```

- Cell 02 `markdown`: `skipped` (0.0 ms)
- Cell 03 `code`: `ok` (1.1 ms)

```text
{'query_id': 'q_incident_001', 'query_raw': 'engine failure after takeoff with emergency return', 'query_normalized': 'engine failure after takeoff with emergency return', 'intent': 'Incident_Report', 'intent_confidence': 0.5491248787888274, 'intent_source': 'ml', 'expanded_queries': ['engine failure after takeoff with emergency return', 'engine failure after takeoff with emergency return aviation incident report', 'engine failure after takeoff with emergency return event narrative', 'engine failure after takeoff with emergency return safety occurrence summary'], 'rewritten_query': 'aviation incident narrative lookup for: engine failure after takeoff with emergency return', 'retrieval_plan': {'strategy': 'semantic', 'fallback_strategy': 'hybrid', 'top_k': 10, 'filters': {}, 'routing_reason': 'Narrative incident queries benefit from semantic similarity over safety reports.'}, 'created_at': '2026-06-06T06:14:28.144450'}
```

- Cell 04 `markdown`: `skipped` (0.0 ms)
- Cell 05 `code`: `ok` (13.0 ms)

```text
{'MiddleAgentOutput': ['query_id', 'predicted_intent', 'topk_docs', 'retrieval_diagnostics', 'created_at'], 'RetrievedDoc': ['doc_id', 'chunk_id', 'chunk_text', 'scores', 'metadata'], 'Required diagnostics for mock': ['adapter_mode=generated_mock', 'contract_owner', 'strategy_requested', 'fallback_strategy']}
                intent              doc_id  ...       source final_score
0      Incident_Report   mock_incident_001  ...  phase2_mock        0.88
1      Incident_Report   mock_incident_002  ...  phase2_mock        0.82
2  Technical_Procedure  mock_procedure_001  ...  phase2_mock        0.91
3  Technical_Procedure  mock_procedure_002  ...  phase2_mock        0.84
4       Metadata_Query   mock_metadata_001  ...  phase2_mock        0.78
5       Metadata_Query   mock_metadata_002  ...  phase2_mock        0.74
6              Factoid    mock_factoid_001  ...  phase2_mock        0.83
7              Factoid    mock_factoid_002  ...  phase2_mock        0.78

[8 rows x 6 columns]
```

- Cell 06 `markdown`: `skipped` (0.0 ms)
- Cell 07 `code`: `ok` (411.5 ms)

```text
query_id     predicted_intent  ... first_doc_source   first_doc_type
0   q_incident_001      Incident_Report  ...     asrs_dataset  incident_report
1  q_procedure_001  Technical_Procedure  ...     asrs_dataset        procedure
2   q_metadata_001       Metadata_Query  ...     asrs_dataset         metadata
3    q_factoid_001              Factoid  ...     asrs_dataset        procedure

[4 rows x 8 columns]
```

- Cell 08 `markdown`: `skipped` (0.0 ms)
- Cell 09 `code`: `ok` (15.9 ms)

```text
query_id    run_mode  ...  hallucination_risk       route_llm
0   q_incident_001  Fast local  ...              0.2403  local fallback
1  q_procedure_001  Fast local  ...              0.2326  local fallback
2   q_metadata_001  Fast local  ...              0.1901  local fallback
3    q_factoid_001  Fast local  ...              0.2101  local fallback

[4 rows x 7 columns]
```

- Cell 10 `markdown`: `skipped` (0.0 ms)
- Cell 11 `code`: `ok` (2.4 ms)

```text
Wrote 4 rows to C:\Users\DELL\Desktop\Vinh Hoang\Master Program\Xử lý ngôn ngữ tự nhiên\Project\artifacts\phase3_hoang_grounded_answer_output.jsonl
Phase 3 contract validation passed.
None
```

- Cell 12 `markdown`: `skipped` (0.0 ms)
- Cell 13 `code`: `ok` (13.1 ms)

```text
query_id  answer_token_count  ...  overlap_ratio  hallucination_risk
0   q_incident_001                 129  ...         0.7597              0.2403
1  q_procedure_001                 129  ...         0.7674              0.2326
2   q_metadata_001                 142  ...         0.8099              0.1901
3    q_factoid_001                 119  ...         0.7899              0.2101

[4 rows x 6 columns]
```

- Cell 14 `markdown`: `skipped` (0.0 ms)
- Cell 15 `code`: `ok` (8.7 ms)

```text
{'sample_size': 4, 'citation_coverage': np.float64(1.0), 'avg_citation_count': np.float64(3.0), 'avg_hallucination_risk': np.float64(0.2183), 'empty_answer_rate': np.float64(0.0)}
          query_id  answer_chars  ...  hallucination_risk  empty_answer
0   q_incident_001          1157  ...              0.2403         False
1  q_procedure_001          1145  ...              0.2326         False
2   q_metadata_001          1162  ...              0.1901         False
3    q_factoid_001          1146  ...              0.2101         False

[4 rows x 6 columns]
```

- Cell 16 `markdown`: `skipped` (0.0 ms)
- Cell 17 `code`: `ok` (13.7 ms)

```text
{'gold_rows': 4, 'pass_rate': 1.0, 'gold_path': 'C:\\Users\\DELL\\Desktop\\Vinh Hoang\\Master Program\\Xử lý ngôn ngữ tự nhiên\\Project\\data\\phase3_grounding_gold_labels.jsonl'}
          query_id  ... passed
0   q_incident_001  ...   True
1  q_procedure_001  ...   True
2   q_metadata_001  ...   True
3    q_factoid_001  ...   True

[4 rows x 6 columns]
```

