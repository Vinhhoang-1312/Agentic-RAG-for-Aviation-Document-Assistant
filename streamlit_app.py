from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import streamlit as st

from aviation_rag.config import Settings, configure_tracing_env, ensure_artifact_dirs
from aviation_rag.graph import build_graph
from aviation_rag.runtime import build_run_state

SAMPLE_QUERIES = [
    "engine failure after takeoff with emergency return",
    "den bao ENG OIL PRESS sang thi lam gi?",
    "crosswind turbulence during final approach at runway 25",
    "what is the meaning of MEL in aviation?",
]

STRATEGIES = ["hybrid", "semantic", "bm25", "metadata_first", "hybrid_rrf"]


@st.cache_resource(show_spinner=False)
def get_runtime(intent_mode: str) -> tuple[Settings, object]:
    settings = replace(Settings(), input_intent_mode=intent_mode)
    ensure_artifact_dirs(settings)
    configure_tracing_env(settings)
    return settings, build_graph(settings)


def run_query(
    *,
    query: str,
    top_k: int,
    strategy: str,
    allow_local_fallback: bool,
    intent_mode: str,
    write_artifacts: bool,
) -> dict[str, Any]:
    settings, graph = get_runtime(intent_mode)
    state = build_run_state(
        settings,
        query_raw=query,
        top_k=top_k,
        strategy=strategy,
        allow_local_fallback=allow_local_fallback,
        write_phase1_artifact=write_artifacts,
        write_phase2_artifact=write_artifacts,
        write_phase3_artifact=write_artifacts,
    )
    return graph.invoke(state)


def score_table(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, doc in enumerate(docs, start=1):
        scores = doc.get("scores", {}) or {}
        metadata = doc.get("metadata", {}) or {}
        rows.append(
            {
                "Rank": index,
                "Doc ID": doc.get("doc_id", "unknown"),
                "Type": metadata.get("document_type", "unknown"),
                "Airport": metadata.get("airport", ""),
                "Semantic": round(float(scores.get("semantic", 0.0)), 4),
                "BM25": round(float(scores.get("bm25", 0.0)), 4),
                "Metadata": round(float(scores.get("metadata", 0.0)), 4),
                "Final": round(float(scores.get("final", 0.0)), 4),
            }
        )
    return rows


def method_badges(diagnostics: dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric("Backend", diagnostics.get("retrieval_backend", "unknown"))
    cols[1].metric("Embedding", diagnostics.get("embedding_backend", "unknown"))
    cols[2].metric("Dim", diagnostics.get("embedding_dim", "n/a"))
    cols[3].metric("Fusion", diagnostics.get("fusion_method", "n/a"))
    latency = diagnostics.get("latency_ms")
    cols[4].metric("Latency", f"{float(latency):.1f} ms" if latency is not None else "n/a")


def show_warnings(diagnostics: dict[str, Any]) -> None:
    adapter_mode = diagnostics.get("adapter_mode")
    embedding_backend = diagnostics.get("embedding_backend")
    fallback_reason = diagnostics.get("fallback_reason")
    if adapter_mode == "generated_mock":
        st.warning("Phase 2 is using generated mock retrieval, not real corpus search.")
    if embedding_backend == "tfidf_svd_faiss_fallback":
        st.warning("Dense MiniLM embeddings were unavailable; semantic search is using explicit TF-IDF/SVD FAISS fallback.")
    if embedding_backend == "unavailable":
        st.error(f"Phase 2 retrieval backend is unavailable: {fallback_reason}")


st.set_page_config(page_title="Aviation RAG Research Demo", page_icon="A", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f4f7f2 0%, #fffaf0 42%, #eaf3f8 100%); }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px; }
    .hero { padding: 1.4rem 1.6rem; border-radius: 18px; background: #102a2d; color: white; box-shadow: 0 18px 55px rgba(16, 42, 45, 0.22); }
    .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -0.03em; }
    .hero p { margin: 0.55rem 0 0 0; color: #dbe9df; font-size: 1rem; }
    .pill { display: inline-block; padding: 0.22rem 0.58rem; border-radius: 999px; background: #e5f0e2; color: #173b2f; font-size: 0.82rem; margin-right: 0.35rem; margin-bottom: 0.25rem; }
    .doc-card { padding: 1rem; border-radius: 14px; background: rgba(255,255,255,0.92); border: 1px solid rgba(15, 23, 42, 0.08); margin-bottom: 0.8rem; }
    .code-note { padding: 0.8rem 1rem; border-left: 4px solid #cc7a29; background: rgba(255, 250, 240, 0.85); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Aviation Document Retrieval System</h1>
      <p>Intent-aware Semantic RAG with Phase 1 routing, complete Phase 2 retrieval, and Phase 3 grounded answer generation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Experiment Controls")
    selected_query = st.selectbox("Sample query", SAMPLE_QUERIES, index=0)
    query = st.text_area("User query", value=selected_query, height=120)
    intent_mode = st.selectbox("Intent mode", ["heuristic", "auto", "ml"], index=0)
    strategy = st.selectbox("Retrieval strategy", STRATEGIES, index=0)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    allow_local_fallback = st.toggle("Allow local fallback answer", value=True)
    write_artifacts = st.toggle("Write artifacts", value=True)
    run = st.button("Run Pipeline", type="primary", use_container_width=True)

    settings_preview = Settings()
    st.caption(f"Dataset: `{settings_preview.data_path.name}`")
    st.caption(f"Embedding model: `{settings_preview.phase2_embedding_model}`")
    st.caption(f"Index dir: `{settings_preview.phase2_index_dir}`")

if run:
    with st.spinner("Running LangGraph workflow with Phase 2 retrieval..."):
        st.session_state["last_result"] = run_query(
            query=query,
            top_k=top_k,
            strategy=strategy,
            allow_local_fallback=allow_local_fallback,
            intent_mode=intent_mode,
            write_artifacts=write_artifacts,
        )

result = st.session_state.get("last_result")

tab_overview, tab_run, tab_compare, tab_evidence, tab_research = st.tabs(
    ["Pipeline Overview", "Run Demo", "Compare Methods", "Evidence & Scores", "Research Notes"]
)

with tab_overview:
    st.subheader("What this screen is for")
    st.write(
        "This demo is the visible control room for the full Aviation RAG pipeline: "
        "Phase 1 classifies intent and chooses a route, Phase 2 retrieves evidence, "
        "and Phase 3 generates a grounded answer from retrieved context."
    )
    c1, c2, c3 = st.columns(3)
    c1.info("Phase 1: TF-IDF + Logistic Regression when ML mode is available, otherwise heuristic routing.")
    c2.info("Phase 2: MiniLM dense embeddings + FAISS, BM25 keyword search, metadata filtering, and hybrid fusion.")
    c3.info("Phase 3: OpenAI grounded QA when configured, with local fallback and hallucination-risk proxy.")
    st.markdown(
        """
        <div class="code-note">
        <strong>How do I know?</strong> Every run returns diagnostics: embedding backend, model name, FAISS index type, chunk count, BM25 status, fusion method, and latency. If a fallback is used, it is shown explicitly.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_run:
    if not result:
        st.info("Run the pipeline from the sidebar to inspect answer, routing, retrieval, and artifacts.")
    else:
        diagnostics = result.get("retrieval_diagnostics", {}) or {}
        show_warnings(diagnostics)
        method_badges(diagnostics)
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Intent", result.get("intent", "unknown"))
        metric_2.metric("Intent Source", result.get("intent_source", "unknown"))
        metric_3.metric("Top Docs", len(result.get("topk_docs", []) or []))
        risk = result.get("hallucination_risk")
        metric_4.metric("Hallucination Risk", f"{float(risk):.3f}" if risk is not None else "n/a")

        st.subheader("Grounded Answer")
        st.write(result.get("answer", ""))
        citations = result.get("citations", []) or []
        if citations:
            st.subheader("Citations")
            for citation in citations:
                st.markdown(
                    f"<span class='pill'>{citation.get('doc_id', 'unknown')}</span> {citation.get('reason', '')}",
                    unsafe_allow_html=True,
                )

        with st.expander("How do I know what ran behind the screen?", expanded=True):
            st.json(diagnostics)

with tab_compare:
    st.subheader("Compare retrieval strategies")
    st.write("Run the same query through BM25, semantic FAISS, weighted hybrid, metadata-first, and RRF hybrid.")
    compare = st.button("Compare all methods", use_container_width=True)
    if compare:
        rows = []
        outputs = {}
        for item in STRATEGIES:
            with st.spinner(f"Running {item}..."):
                output = run_query(
                    query=query,
                    top_k=top_k,
                    strategy=item,
                    allow_local_fallback=allow_local_fallback,
                    intent_mode=intent_mode,
                    write_artifacts=False,
                )
            outputs[item] = output
            docs = output.get("topk_docs", []) or []
            diagnostics = output.get("retrieval_diagnostics", {}) or {}
            rows.append(
                {
                    "Strategy": item,
                    "Top Doc": docs[0].get("doc_id", "n/a") if docs else "n/a",
                    "Top Type": (docs[0].get("metadata", {}) or {}).get("document_type", "n/a") if docs else "n/a",
                    "Backend": diagnostics.get("embedding_backend", "unknown"),
                    "Fusion": diagnostics.get("fusion_method", "n/a"),
                    "Latency ms": round(float(diagnostics.get("latency_ms", 0.0)), 1),
                    "Risk": round(float(output.get("hallucination_risk", 0.0)), 3),
                }
            )
        st.session_state["compare_outputs"] = outputs
        st.table(rows)

    outputs = st.session_state.get("compare_outputs", {})
    if outputs:
        selected = st.selectbox("Inspect method", list(outputs.keys()))
        selected_output = outputs[selected]
        st.table(score_table(selected_output.get("topk_docs", []) or []))
        st.json(selected_output.get("retrieval_diagnostics", {}) or {})

with tab_evidence:
    if not result:
        st.info("Run a query first to see retrieved evidence and score decomposition.")
    else:
        docs = result.get("topk_docs", []) or []
        st.subheader("Score decomposition")
        st.table(score_table(docs))
        for index, doc in enumerate(docs, start=1):
            metadata = doc.get("metadata", {}) or {}
            scores = doc.get("scores", {}) or {}
            st.markdown(
                f"""
                <div class="doc-card">
                  <strong>{index}. {doc.get('doc_id', 'unknown')}</strong><br/>
                  <span class="pill">{metadata.get('document_type', 'unknown')}</span>
                  <span class="pill">airport: {metadata.get('airport', '') or 'n/a'}</span>
                  <span class="pill">final: {float(scores.get('final', 0.0)):.4f}</span>
                  <p>{doc.get('chunk_text', '')}</p>
                  <code>{json.dumps(scores, ensure_ascii=False)}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_research:
    st.subheader("Notebook to app handoff")
    st.write(
        "The two notebooks are research artifacts. The app operationalizes them: "
        "Phase 1 notebook explores intent routing; Phase 3 notebook explores grounded answer generation. "
        "The new Phase 2 modules now complete the missing retrieval research story."
    )
    st.markdown("""
    - `notebooks/phase1_hoang_intent_routing_research.ipynb`: intent labels, query normalization, expansion, routing policy.
    - `aviation_rag/phase2_indexing.py`: shared corpus preparation, chunking, dedupe, metadata extraction.
    - `aviation_rag/phase2_retrieval_engine.py`: MiniLM/FAISS semantic retrieval, BM25, metadata filters, hybrid/RRF fusion.
    - `notebooks/phase3_hoang_grounded_output_research.ipynb`: grounded QA, citations, hallucination-risk proxy.
    """)
    st.subheader("Artifact paths")
    settings, _graph = get_runtime(intent_mode)
    st.write(f"Phase 1: `{settings.phase1_output_path}`")
    st.write(f"Phase 2: `{settings.phase2_output_path}`")
    st.write(f"Phase 2 index: `{settings.phase2_index_dir}`")
    st.write(f"Phase 3: `{settings.phase3_output_path}`")
