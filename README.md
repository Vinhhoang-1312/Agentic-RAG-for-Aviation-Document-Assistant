# Aviation Document Retrieval System
## Intent-Aware Semantic RAG for Aviation Safety

This repository implements a full **Retrieval-Augmented Generation (RAG)** pipeline for aviation safety documents, built as a final project for the NLP course.

The system uses a **3-phase agentic architecture** orchestrated by LangGraph:

1. **Phase 1 — Intent-Aware Routing** (Vinh Hoang)
2. **Phase 2 — Semantic Retrieval** (Quan San)
3. **Phase 3 — Grounded QA Generation** (Vinh Hoang)

---

## Team Ownership

### Vinh Hoang
- **Phase 1**: Intent classification (TF-IDF + Logistic Regression / heuristic), query normalization, expansion, rewriting, dynamic retrieval routing
- **Phase 3**: Grounded answer generation (OpenAI GPT), citation attachment, hallucination-risk estimation
- **Orchestration**: LangGraph workflow, CLI, API, LangSmith tracing

### Quan San
- **Phase 2**: Semantic Retrieval Engine with 4 strategies
  - `semantic` — FAISS cosine similarity (all-MiniLM-L6-v2, 384-dim)
  - `bm25` — BM25Okapi keyword scoring
  - `hybrid` — Reciprocal Rank Fusion (semantic + BM25)
  - `metadata_first` — Metadata filtering → semantic search on subset
- Text preprocessing, chunking, embedding, indexing
- Producing real retrieval results from ASRS aviation incident reports

### Shared
- `LangGraph` workflow structure
- `LangSmith` tracing
- JSONL artifact contracts (`schemas.py`)
- CLI / API integration

---

## System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Phase 1 — Intent-Aware Routing     │  (Vinh Hoang)
│  • Intent Classification            │
│  • Query Expansion & Rewriting      │
│  • Retrieval Strategy Selection     │
└──────────────┬──────────────────────┘
               │ InputAgentOutput
               ▼
┌─────────────────────────────────────┐
│  Phase 2 — Semantic Retrieval       │  (Quan San)
│  • FAISS / BM25 / Hybrid / Meta    │
│  • 6,745+ chunks from ASRS data    │
│  • ~15–47ms per query              │
└──────────────┬──────────────────────┘
               │ MiddleAgentOutput
               ▼
┌─────────────────────────────────────┐
│  Phase 3 — Grounded QA             │  (Vinh Hoang)
│  • LLM Answer Generation           │
│  • Citation & Hallucination Check   │
└─────────────────────────────────────┘
               │
               ▼
         Grounded Answer
```

---

## Folder Map

```
Agentic-RAG-for-Aviation-Document-Assistant/
├── aviation_rag/                     # Main Python package
│   ├── __init__.py
│   ├── config.py                     # Central settings (paths, model, env)
│   ├── schemas.py                    # Shared Pydantic contracts
│   ├── io_utils.py                   # JSONL read/write utilities
│   ├── intent_rules.py              # Intent label mapping rules
│   ├── graph.py                      # LangGraph workflow (Phase 1→2→3)
│   ├── runtime.py                    # State builder for CLI/API
│   │
│   ├── phase1_hoang_intent_routing.py   # Phase 1 (Hoang)
│   ├── phase2_san_contract_adapter.py   # Phase 2 adapter (San)
│   ├── phase3_hoang_grounded_qa.py      # Phase 3 (Hoang)
│   │
│   ├── retrieval/                    # Retrieval engine sub-package (San)
│   │   ├── __init__.py
│   │   ├── engine.py                 # Core engine — 4 strategies
│   │   ├── indexer.py                # FAISS + BM25 index builder
│   │   └── preprocess.py            # Text normalization & chunking
│   │
│   ├── api.py                        # FastAPI HTTP endpoint
│   ├── cli.py                        # Single-query CLI
│   └── chat_cli.py                   # Interactive chat CLI
│
├── scripts/
│   ├── run_phase1_hoang_intent_routing.py
│   ├── build_phase2_san_index.py     # Build retrieval index (San)
│   ├── validate_phase2_san_contract.py
│   └── evaluate_phase3_hoang_grounding.py
│
├── tests/
│   ├── test_retrieval_engine.py      # Phase 2 engine tests (San)
│   ├── test_phase2_contract_adapter.py
│   ├── test_contracts.py
│   ├── test_pipeline_smoke.py
│   ├── test_input_agent_modes.py
│   ├── test_output_agent_fallback.py
│   ├── test_api_smoke.py
│   └── test_runtime_state.py
│
├── notebooks/
│   ├── phase1_hoang_intent_routing_research.ipynb
│   └── phase3_hoang_grounded_output_research.ipynb
│
├── artifacts/                        # JSONL artifacts (gitignored except samples)
│   ├── phase0_user_query_samples.sample.jsonl
│   └── phase2_san_retrieval_output.sample.jsonl
│
├── data/                             # Local-only data (gitignored)
│   ├── kaggle/ASRS-clean-dataset-aviation-safety.csv
│   └── index_store/                  # FAISS + BM25 index (built once)
│
├── app.py                            # Streamlit Demo UI
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the retrieval index (Phase 2 — Quan San)

```bash
# Full dataset (111K records, ~30 min, recommended for final demo)
python scripts/build_phase2_san_index.py

# Quick sample for development (5K records, ~3 min)
python scripts/build_phase2_san_index.py --sample 5000

# Force rebuild
python scripts/build_phase2_san_index.py --force
```

> **Note:** The index is built once and saved to `data/index_store/`. Subsequent runs load from disk (~1-2s).

### 3. Run the pipeline
Quick local health check after starting the demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_demo.ps1
```

Health:

```bash
# Streamlit Demo UI (recommended for demo)
streamlit run app.py

# Single query via CLI
python -m aviation_rag.cli run "engine failure after takeoff"

# Interactive chat
python -m aviation_rag.chat_cli

# HTTP API
uvicorn aviation_rag.api:app --host 0.0.0.0 --port 8000
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

---

## Artifact Contracts

### Phase 1 → Phase 2: `InputAgentOutput`

File: `artifacts/phase1_hoang_intent_routing_output.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | str | Unique query identifier |
| `query_raw` | str | Original user query |
| `query_normalized` | str | Cleaned, jargon-expanded query |
| `intent` | enum | `Incident_Report` / `Technical_Procedure` / `Metadata_Query` / `Factoid` |
| `intent_confidence` | float | 0.0–1.0 |
| `intent_source` | str | `ml` or `heuristic` |
| `expanded_queries` | list | Intent-aware query expansions |
| `rewritten_query` | str | Rewritten for retrieval optimization |
| `retrieval_plan` | object | `{strategy, fallback_strategy, top_k, filters, routing_reason}` |

### Phase 2 → Phase 3: `MiddleAgentOutput`
### Research mode
- Trains `TF-IDF + Logistic Regression` on query-only corpus (seed + training JSONL + augmentation)
- Gold-set (`data/phase1_intent_gold_labels.jsonl`) is evaluation-only
- Useful for notebook experiments

### App runtime mode
- Default intent routing: `auto` (ML + heuristic fallback)
- Uses saved model in `artifacts/phase1_intent_model/` (auto-retrains if stale ASRS-narrative model detected)
- Uses local FAISS retrieval over the ASRS dataset when local data is available
- Still works without San retrieval engine by falling back to the Phase 2 contract adapter
- Uses lazy FastAPI runtime initialization so `/health` responds quickly and the heavier graph setup happens only on first real chat request

Key environment variables:

```bash
set INPUT_INTENT_MODE=auto
set PHASE1_ML_CONFIDENCE_THRESHOLD=0.55
set PHASE1_RETRAIN=false
```

Modes:
- `auto`: recommended; ML when confident/agrees with heuristic, else heuristic
- `ml`: always use TF-IDF + Logistic Regression
- `heuristic`: rule-based routing only (no ML at inference)

Retrieval environment variables:

```bash
set RETRIEVAL_MAX_DOCS=15000
set RETRIEVAL_TFIDF_MAX_FEATURES=12000
set RETRIEVAL_SVD_COMPONENTS=128
```

File: `artifacts/phase2_san_retrieval_output.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | str | Same as Phase 1 |
| `predicted_intent` | enum | Intent label |
| `topk_docs` | list | Retrieved documents (see below) |
| `retrieval_diagnostics` | dict | Strategy used, timing, index size |

Each `topk_docs` item (`RetrievedDoc`):
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
  Purpose: legacy keyword helper retained for compatibility; not the main runtime classifier.
- `IntentModel.predict(text)`
  Purpose: return predicted label and confidence from TF-IDF + Logistic Regression.
- `Phase1HoangIntentRouting._train_intent_model(data_path)`
  Purpose: train the required TF-IDF + Logistic Regression classifier from seed examples plus local ASRS weak labels.
- `Phase1HoangIntentRouting.predict_intent(query_raw)`
  Purpose: predict final intent, confidence, and source from the sklearn classifier.
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
- `Phase3HoangGroundedQA._call_route_llm(question, context_block, doc_ids)`
  Purpose: grounded answer generation through OpenRouter model queue.
- `Phase3HoangGroundedQA._call_route_llm_model(...)`
  Purpose: call one OpenRouter model; Phase 3 retries once, then moves to the next model.
- `Phase3HoangGroundedQA._fallback_answer(question, middle_output)`
  Purpose: offline/local fallback answer.
- `Phase3HoangGroundedQA._grounding_metrics(answer, contexts)`
  Purpose: compute overlap-based hallucination proxy.
- `Phase3HoangGroundedQA.generate(question, middle_output, allow_fallback)`
  Purpose: create final grounded answer object.
- `Phase3HoangGroundedQA.write_output(output, path)`
  Purpose: write Phase 3 artifact row.

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | str | e.g., `asrs_1314306` |
| `chunk_id` | str | e.g., `asrs_1314306#0` |
| `chunk_text` | str | Retrieved text chunk |
| `scores` | dict | `{semantic, bm25, hybrid, rrf, final}` |
| `metadata` | dict | ASRS fields: event_id, aircraft, phase, etc. |

### Phase 3 Output: `FinalOutput`

File: `artifacts/phase3_hoang_grounded_answer_output.jsonl`
- `scripts/run_phase1_hoang_intent_routing.py`
  Purpose: generate Phase 1 artifact from raw query input.
- `scripts/validate_phase2_san_contract.py`
  Purpose: validate San's Phase 2 artifact against shared schema.
- `scripts/evaluate_phase3_hoang_grounding.py`
  Purpose: summarize grounding quality from Phase 3 artifact.
- `scripts/start_demo.ps1`
  Purpose: open local API and Streamlit demo windows for Windows users.
- `scripts/start_api.ps1`
  Purpose: run only the FastAPI server.
- `scripts/start_streamlit.ps1`
  Purpose: run only the Streamlit demo UI.
- `scripts/check_demo.ps1`
  Purpose: verify local API and UI health endpoints quickly.

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | str | Same as Phase 1 |
| `answer` | str | Grounded answer from LLM |
| `citations` | list | `[{doc_id, chunk_id, reason}]` |
| `hallucination_risk` | float | 0.0–1.0 |
| `grounding_report` | dict | Overlap metrics |

---

## Intent Routing Policy

Phase 1 classifies queries into 4 intents and routes to the appropriate retrieval strategy:

| Intent | Strategy | Fallback | Routing Reason |
|--------|----------|----------|----------------|
| `Incident_Report` | `semantic` | `hybrid` | Narrative queries → semantic similarity over safety reports |
| `Technical_Procedure` | `bm25` | `hybrid` | Procedure queries → keyword-heavy checklist/manual retrieval |
| `Metadata_Query` | `metadata_first` | `bm25` | Metadata queries → filter structured fields first |
| `Factoid` | `semantic` | `hybrid` | Factoid queries → concise semantic lookup |

---

## Phase 2 — Semantic Retrieval Details (Quan San)

### Retrieval Strategies

| Strategy | Method | Performance |
|----------|--------|-------------|
| **semantic** | FAISS `IndexFlatIP` cosine search on L2-normalized embeddings | ~11–15ms |
| **bm25** | BM25Okapi keyword scoring, normalized to [0,1] | ~30ms |
| **hybrid** | Reciprocal Rank Fusion (k=60), weighted 0.7 semantic + 0.3 BM25 | ~40ms |
| **metadata_first** | Filter metadata → semantic search on subset → fallback if <top_k matches | ~47ms |

### Technical Specifications

| Spec | Value |
|------|-------|
| Embedding model | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| Dimension | 384 |
| Normalization | L2 (unit vectors for cosine via inner product) |
| FAISS index | `IndexFlatIP` (exact search) |
| Chunking | 512 words max, 50 words overlap, sentence-boundary aware |
| Data source | ASRS Clean Dataset (111,492 records, 63 columns) |

### Fallback Chain

The Phase 2 adapter resolves output with graceful fallback:

1. **Cached output** → return if `query_id` already in artifact file
2. **Real retrieval** → FAISS/BM25 search if index is built ✅
3. **Sample artifact** → static sample JSONL if available
4. **Mock fallback** → generated placeholder (pipeline never crashes)

---

## Research vs App Mode

### Research mode
- Uses local dataset in `data/` for ML intent classification
- Useful for notebook experiments

### App runtime mode
- Works without dataset (heuristic fallback)
- Works without retrieval index (mock fallback)

```bash
# Environment variable
set INPUT_INTENT_MODE=heuristic   # default, no dataset needed
set INPUT_INTENT_MODE=auto        # ML when possible, else heuristic
set INPUT_INTENT_MODE=ml          # force ML (needs dataset)
```

---

## LangSmith / LangGraph

LangGraph is the local orchestration engine. LangSmith is optional observability.

```bash
# Default (tracing off)
set LANGSMITH_TRACING=false

# Enable tracing
set LANGSMITH_TRACING=true
set LANGSMITH_API_KEY=...
set LANGSMITH_PROJECT=aviation-rag-team
```

---

## Function Catalog

### `aviation_rag/config.py`
- `Settings` — Central runtime paths and environment configuration
- `ensure_artifact_dirs()` — Create artifact directory
- `configure_tracing_env()` — Normalize LangSmith env

### `aviation_rag/schemas.py`
- `RetrievalPlan` — Routing decision from Phase 1
- `InputAgentOutput` — Contract: Phase 1 → Phase 2
- `RetrievedDoc` — Single retrieved chunk
- `MiddleAgentOutput` — Contract: Phase 2 → Phase 3
- `Citation` — Citation entry in final answer
- `FinalOutput` — Final grounded answer

### `aviation_rag/phase1_hoang_intent_routing.py` (Hoang)
- `Phase1HoangIntentRouting` — Intent classification + routing
- `normalize_text()`, `heuristic_intent()`, `IntentModel`

### `aviation_rag/retrieval/engine.py` (San)
- `RetrievalEngine` — Core engine with 4 strategies
- `retrieve(input_row)` — Main entry: `InputAgentOutput` → `MiddleAgentOutput`
- `_search_semantic()`, `_search_bm25()`, `_search_hybrid()`, `_search_metadata_first()`

### `aviation_rag/retrieval/indexer.py` (San)
- `build_and_save_index()` — Full build: CSV → preprocess → embed → FAISS + BM25 → disk
- `load_index()` — Load from disk
- `index_exists()` — Check existence

### `aviation_rag/retrieval/preprocess.py` (San)
- `normalize_text()` — Lowercase, URL removal, whitespace
- `combine_text_fields()` — Merge [SUMMARY] + [REPORT 1] + [REPORT 2]
- `chunk_text()` — Sentence-boundary-aware chunking
- `load_and_preprocess()` — Full CSV → (chunks, metadata) pipeline

### `aviation_rag/phase2_san_contract_adapter.py` (San)
- `Phase2SanContractAdapter` — Adapter with fallback chain
- `resolve_output()` — cached → real engine → sample → mock
- `write_output()` — Save to JSONL

### `aviation_rag/phase3_hoang_grounded_qa.py` (Hoang)
- `Phase3HoangGroundedQA` — Grounded answer generation
- `generate()` — LLM answer + citations + hallucination check

### `aviation_rag/graph.py`
- `RagState` — LangGraph state container
- `build_graph()` — Assemble Phase 1 → 2 → 3

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_phase1_hoang_intent_routing.py` | Generate Phase 1 artifact |
| `scripts/build_phase2_san_index.py` | Build FAISS + BM25 index (San) |
| `scripts/validate_phase2_san_contract.py` | Validate Phase 2 artifact schema |
| `scripts/evaluate_phase3_hoang_grounding.py` | Evaluate grounding quality |

---

## Test Command

```bash
python -m pytest tests/ -v
```

---

## Minimal Files to Read First

1. `README.md`
2. `aviation_rag/schemas.py` — data contracts
3. `aviation_rag/graph.py` — pipeline orchestration
4. `aviation_rag/phase1_hoang_intent_routing.py` — Phase 1
5. `aviation_rag/retrieval/engine.py` — Phase 2
6. `aviation_rag/phase3_hoang_grounded_qa.py` — Phase 3
