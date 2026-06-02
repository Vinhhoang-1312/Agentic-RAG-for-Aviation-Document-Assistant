# Vinh Hoang Phase-Based Aviation Workflow

This repository contains Vinh Hoang's phase ownership plus a local integrated demo path for the full aviation RAG project:

- **Phase 1 - Intent-Aware Routing**
- **Phase 3 - Grounded Output**

`Quang San` will still be able to plug in his own Phase 2 artifact.  
This repo now supports both:
- a **real local FAISS retrieval path** over the ASRS dataset for integrated demos
- a **Phase 2 contract adapter + generated mock fallback** when retrieval data is unavailable

## What This Repo Owns

### Vinh Hoang owns
- Phase 1: user query understanding
- Intent classification
- Query normalization
- Query expansion
- Query rewriting
- Dynamic retrieval routing
- Exporting the shared artifact for San
- Phase 3 grounded answer generation
- Citation attachment
- Hallucination-risk estimation
- LangGraph orchestration around phases 1 -> 2 contract -> 3

### Quang San owns
- The shared Phase 2 output contract
- Partner-side retrieval experimentation and alternative retrieval implementations
- Producing a real Phase 2 artifact that matches the shared contract

### Shared
- `LangGraph` workflow structure
- `LangSmith` tracing
- JSONL artifact contracts
- CLI / API integration shape

## Full Flow Explained

### Real team flow
1. User sends a query.
2. **Phase 1 - Hoang** classifies intent and selects retrieval strategy.
3. Hoang exports `phase1_hoang_intent_routing_output.jsonl`.
4. **Phase 2 - San** reads that file and performs retrieval.
5. San exports `phase2_san_retrieval_output.jsonl`.
6. **Phase 3 - Hoang** reads San's output and generates grounded answer.
7. Hoang exports `phase3_hoang_grounded_answer_output.jsonl`.

### Local demo flow in this repo
1. User sends a query.
2. Hoang Phase 1 runs normally.
3. If San's Phase 2 artifact exists, LangGraph reads it.
4. If San's artifact is missing, the local Phase 2 runtime first attempts FAISS retrieval over the local dataset.
5. If retrieval data is unavailable, the Phase 2 contract adapter uses generated mock data.
6. Hoang Phase 3 still runs end-to-end for local demo and testing.

## Folder Map

- `aviation_rag/`
  Runtime package for LangGraph, CLI, API, shared schemas, and Hoang-owned phase code.
- `notebooks/`
  Hoang research notebooks only.
- `scripts/`
  Utility scripts for Phase 1 generation, Phase 2 contract validation, and Phase 3 evaluation.
- `tests/`
  Contract, smoke, and runtime tests. Keep this folder because it protects the API, graph, retrieval, and schema behavior during refactors.
- `artifacts/`
  Runtime output artifacts and retained placeholders such as `.gitkeep`.
- `data/`
  Local-only research data. Useful for Phase 1 research mode, not required for app runtime.

## Artifact Contracts

### Phase 1 output
File:
- `artifacts/phase1_hoang_intent_routing_output.jsonl`

Produced by:
- Hoang Phase 1 notebook
- Hoang CLI / API / LangGraph runtime

Schema:
- `query_id`
- `query_raw`
- `query_normalized`
- `intent`
- `intent_confidence`
- `intent_source`
- `expanded_queries`
- `rewritten_query`
- `retrieval_plan`
- `created_at`

`retrieval_plan` fields:
- `strategy`
- `fallback_strategy`
- `top_k`
- `filters`
- `routing_reason`

### Phase 2 output
File:
- `artifacts/phase2_san_retrieval_output.jsonl`

Owned by:
- Quang San

Schema:
- `query_id`
- `predicted_intent`
- `topk_docs`
- `retrieval_diagnostics`
- `created_at`

Each `topk_docs` item contains:
- `doc_id`
- `chunk_id`
- `chunk_text`
- `scores`
- `metadata`

### Phase 3 output
File:
- `artifacts/phase3_hoang_grounded_answer_output.jsonl`

Produced by:
- Hoang Phase 3 notebook
- Hoang CLI / API / LangGraph runtime

Schema:
- `query_id`
- `answer`
- `citations`
- `hallucination_risk`
- `grounding_report`
- `created_at`

## Intent Taxonomy Used by Hoang

Hoang Phase 1 routes into 4 intents:

- `Incident_Report`
- `Technical_Procedure`
- `Metadata_Query`
- `Factoid`

Default routing policy:

- `Incident_Report -> semantic`, fallback `hybrid`
- `Technical_Procedure -> bm25`, fallback `hybrid`
- `Metadata_Query -> metadata_first`, fallback `bm25`
- `Factoid -> semantic`, fallback `hybrid`

Important note:
- `Factoid` is mainly handled by runtime heuristics.
- Research mode may train on local dataset for the other labels when data is available.

## Notebook Map

### `notebooks/phase1_hoang_intent_routing_research.ipynb`
Purpose:
- Hoang's research notebook for the full pre-retrieval phase.

Cell map:
- Cell 0: title and scope of Hoang phase 1
- Cell 1: imports and settings overview
- Cell 2: sample-query explanation
- Cell 3: sample query list for the 4 intents
- Cell 4: section header for intent classification
- Cell 5: build Phase 1 outputs and inspect intent / rewritten query
- Cell 6: section header for dynamic routing
- Cell 7: inspect strategy, fallback strategy, filters, and routing reason
- Cell 8: section header for export
- Cell 9: write `phase1_hoang_intent_routing_output.jsonl` and validate schema

### `notebooks/phase3_hoang_grounded_output_research.ipynb`
Purpose:
- Hoang's research notebook for grounded output generation.

Cell map:
- Cell 0: title and scope of Hoang phase 3
- Cell 1: imports and artifact path overview
- Cell 2: section header for loading phase 1 artifact
- Cell 3: load Phase 1 artifact and inspect first row
- Cell 4: section header for Phase 2 contract consumption
- Cell 5: resolve San contract via real artifact, local FAISS retrieval, or generated mock fallback
- Cell 6: section header for grounded QA generation
- Cell 7: generate final grounded outputs
- Cell 8: section header for export
- Cell 9: write `phase3_hoang_grounded_answer_output.jsonl` and validate schema

## Runtime Entry Points

### CLI
Command:

```bash
python -m aviation_rag.cli --query "engine failure after takeoff" --write-phase1-artifact
```

What it does:
- Runs Phase 1
- Resolves Phase 2 via San artifact, local FAISS retrieval, or generated mock fallback
- Runs Phase 3
- Writes phase-based artifacts

### Interactive chat
Command:

```bash
python -m aviation_rag.chat_cli
```

### Streamlit demo UI
Command:

```bash
streamlit run streamlit_app.py
```

Recommended Windows launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1
```

### HTTP API
Command:

```bash
uvicorn aviation_rag.api:app --host 0.0.0.0 --port 8000
```

Quick local health check after starting the demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_demo.ps1
```

Health:

```bash
curl http://localhost:8000/health
```

Chat:

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"engine failure after takeoff\",\"strategy\":\"semantic\",\"top_k\":10}"
```

## Research vs App Mode

### Research mode
- Can use local dataset in `data/`
- Can train `TF-IDF + Logistic Regression` for intent classification
- Useful for notebook experiments

### App runtime mode
- Works without requiring dataset training at request time
- Uses heuristic fallback for intent classification when needed
- Uses local FAISS retrieval over the ASRS dataset when local data is available
- Still works without San retrieval engine by falling back to the Phase 2 contract adapter

Key environment variable:

```bash
set INPUT_INTENT_MODE=heuristic
```

Modes:
- `heuristic`: default app mode, query-only and no dataset dependency at runtime
- `auto`: use ML when possible, else heuristic
- `ml`: force dataset-backed ML path

Retrieval environment variables:

```bash
set RETRIEVAL_MAX_DOCS=15000
set RETRIEVAL_TFIDF_MAX_FEATURES=12000
set RETRIEVAL_SVD_COMPONENTS=128
```

## Function Catalog

### `aviation_rag/config.py`
- `Settings`
  Purpose: central runtime paths and environment configuration.
  Inputs: `.env` values and project-relative defaults.
  Outputs: immutable config object used by CLI, API, notebooks, and tests.
- `ensure_artifact_dirs(settings)`
  Purpose: create artifact directory if missing.
  Inputs: `Settings`.
  Outputs: filesystem side effect only.
- `configure_tracing_env(settings)`
  Purpose: normalize LangSmith tracing env for stable local runs.
  Inputs: `Settings.langsmith_*`.
  Outputs: process env mutation.

### `aviation_rag/schemas.py`
- `RetrievalPlan`
  Purpose: carry Hoang's routing decision toward Phase 2.
- `InputAgentOutput`
  Purpose: contract from Hoang Phase 1 to San Phase 2.
- `RetrievedDoc`
  Purpose: schema for a retrieved chunk row.
- `MiddleAgentOutput`
  Purpose: contract from San Phase 2 to Hoang Phase 3.
- `Citation`
  Purpose: citation entry attached to final answer.
- `FinalOutput`
  Purpose: final grounded answer artifact.

### `aviation_rag/io_utils.py`
- `append_jsonl(path, row)`
  Purpose: append one row to JSONL file.
- `write_jsonl(path, rows)`
  Purpose: overwrite a JSONL file with rows.
- `read_jsonl(path)`
  Purpose: read entire JSONL file into memory.
- `find_by_query_id(path, query_id)`
  Purpose: locate one row by `query_id`.

### `aviation_rag/intent_rules.py`
- `map_row_to_intent(row)`
  Purpose: map dataset rows to Hoang Phase 1 training labels.
  Inputs: local dataframe row.
  Outputs: `Incident_Report`, `Technical_Procedure`, or `Metadata_Query`.

### `aviation_rag/phase1_hoang_intent_routing.py`
- `normalize_text(text)`
  Purpose: normalize aviation query text and expand jargon.
- `tokenize(text)`
  Purpose: token extraction helper.
- `heuristic_intent(normalized_query)`
  Purpose: query-only intent fallback, including `Factoid`.
- `IntentModel.predict(text)`
  Purpose: return predicted label and confidence from TF-IDF + Logistic Regression.
- `Phase1HoangIntentRouting._maybe_train_intent_model(data_path)`
  Purpose: optional ML training path for research mode.
- `Phase1HoangIntentRouting.predict_intent(query_raw)`
  Purpose: choose final intent, confidence, and source (`ml` or `heuristic`).
- `Phase1HoangIntentRouting.expand_query(normalized_query, intent)`
  Purpose: build intent-aware query expansion set.
- `Phase1HoangIntentRouting.rewrite_query(query_raw, intent)`
  Purpose: build rewritten query tailored to retrieval.
- `Phase1HoangIntentRouting.build_retrieval_plan(intent, top_k)`
  Purpose: attach routing strategy, fallback, filters, and routing reason.
- `Phase1HoangIntentRouting.build_output(...)`
  Purpose: create one Phase 1 contract row.
- `Phase1HoangIntentRouting.write_output(output, path)`
  Purpose: write one Phase 1 output row.
- `Phase1HoangIntentRouting.load_output(query_id, path)`
  Purpose: load one Phase 1 row by id.

### `aviation_rag/phase2_san_contract_adapter.py`
- `Phase2SanContractAdapter._build_mock_output(input_row)`
  Purpose: synthesize a local demo Phase 2 artifact without owning retrieval logic.
- `Phase2SanContractAdapter.resolve_output(input_row, output_path)`
  Purpose: prefer San artifact, else generated mock.
- `Phase2SanContractAdapter.write_output(output, path)`
  Purpose: materialize resolved Phase 2 output for debugging/demo.

### `aviation_rag/phase2_san_faiss_retrieval.py`
- `Phase2SanFaissRetrieval.retrieve(input_row)`
  Purpose: run local FAISS retrieval over the ASRS dataset with TF-IDF/SVD semantic vectors, BM25 lexical scoring, and intent-aware ranking.

### `aviation_rag/phase3_hoang_grounded_qa.py`
- `_tokenize(text)`
  Purpose: token overlap helper for grounding metrics.
- `Phase3HoangGroundedQA._build_context(middle_output)`
  Purpose: turn Phase 2 rows into LLM context block.
- `Phase3HoangGroundedQA._call_openai(question, context_block, doc_ids)`
  Purpose: grounded answer generation with OpenAI.
- `Phase3HoangGroundedQA._fallback_answer(question, middle_output)`
  Purpose: offline/local fallback answer.
- `Phase3HoangGroundedQA._grounding_metrics(answer, contexts)`
  Purpose: compute overlap-based hallucination proxy.
- `Phase3HoangGroundedQA.generate(question, middle_output, allow_fallback)`
  Purpose: create final grounded answer object.
- `Phase3HoangGroundedQA.write_output(output, path)`
  Purpose: write Phase 3 artifact row.

### `aviation_rag/graph.py`
- `RagState`
  Purpose: state container passed between LangGraph nodes.
- `build_graph(settings)`
  Purpose: assemble Hoang Phase 1 -> San contract node -> Hoang Phase 3.
  Node order:
  - `phase1_hoang_input_node`
  - `phase2_san_contract_node`
  - `phase3_hoang_output_node`

### `aviation_rag/runtime.py`
- `build_run_state(...)`
  Purpose: create state payload for CLI, chat CLI, and API entrypoints.

### `aviation_rag/cli.py`
- `build_parser()`
  Purpose: define command-line interface for end-to-end run.
- `main()`
  Purpose: execute LangGraph run and print summary JSON.

### `aviation_rag/chat_cli.py`
- `_now_iso()`
  Purpose: timestamp helper.
- `build_parser()`
  Purpose: CLI options for interactive chat.
- `_session_path(settings, session_id)`
  Purpose: resolve chat log file path.
- `_print_assistant(result)`
  Purpose: pretty-print answer + diagnostics.
- `main()`
  Purpose: interactive end-to-end phase workflow.

### `aviation_rag/api.py`
- `ChatRequest`
  Purpose: input schema for `/v1/chat`.
- `ChatResponse`
  Purpose: output schema for `/v1/chat`.
- `create_app(settings=None)`
  Purpose: construct FastAPI app with LangGraph runtime.

## Scripts

- `scripts/run_phase1_hoang_intent_routing.py`
  Purpose: generate Phase 1 artifact from raw query input.
- `scripts/validate_phase2_san_contract.py`
  Purpose: validate San's Phase 2 artifact against shared schema.
- `scripts/evaluate_phase3_hoang_grounding.py`
  Purpose: summarize grounding quality from Phase 3 artifact.

## San Handoff Contract

San only needs these files from Hoang:

- `artifacts/phase1_hoang_intent_routing_output.jsonl`
- `README.md`
- `aviation_rag/schemas.py`

San must return:

- `artifacts/phase2_san_retrieval_output.jsonl`

San output must contain:
- same `query_id` from Hoang Phase 1
- `predicted_intent`
- `topk_docs`
- `retrieval_diagnostics`

LangGraph in this repo will consume San's output automatically through:
- `aviation_rag/phase2_san_contract_adapter.py`

## LangSmith / LangGraph

- `LangGraph` is the local orchestration engine in code.
- `LangSmith` is optional observability.
- Local runs should stay stable with tracing off.

Recommended local default:

```bash
set LANGSMITH_TRACING=false
set LANGCHAIN_TRACING_V2=false
```

Enable tracing only when needed:

```bash
set LANGSMITH_TRACING=true
set LANGCHAIN_TRACING_V2=true
set LANGSMITH_API_KEY=...
set LANGSMITH_PROJECT=aviation-rag-team
```

## Minimal Files a Future Agent Should Read First

If a future agent joins this project, read these files first and only then expand outward:

1. `README.md`
2. `aviation_rag/graph.py`
3. `aviation_rag/phase1_hoang_intent_routing.py`
4. `aviation_rag/phase2_san_faiss_retrieval.py`
5. `aviation_rag/phase2_san_contract_adapter.py`
6. `aviation_rag/phase3_hoang_grounded_qa.py`
7. `aviation_rag/schemas.py`

That is enough to understand:
- ownership
- data flow
- notebook to app handoff
- artifact contracts
- LangGraph orchestration

## Test Command

```bash
python -m unittest discover -s tests -q
```
