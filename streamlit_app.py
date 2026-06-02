from __future__ import annotations

import json

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


@st.cache_resource(show_spinner=False)
def get_runtime() -> tuple[Settings, object]:
    settings = Settings()
    ensure_artifact_dirs(settings)
    configure_tracing_env(settings)
    return settings, build_graph(settings)


def run_query(query: str, top_k: int, strategy: str, allow_local_fallback: bool) -> dict:
    settings, graph = get_runtime()
    state = build_run_state(
        settings,
        query_raw=query,
        top_k=top_k,
        strategy=strategy,
        allow_local_fallback=allow_local_fallback,
        write_phase1_artifact=True,
        write_phase2_artifact=True,
        write_phase3_artifact=True,
    )
    return graph.invoke(state)


st.set_page_config(
    page_title="Aviation Retrieval Demo",
    page_icon="A",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f3f7fb 0%, #ffffff 32%, #eef5ef 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        margin: 0;
        color: #0f172a;
        font-size: 2.1rem;
    }
    .hero p {
        margin: 0.5rem 0 0 0;
        color: #334155;
        font-size: 1rem;
    }
    .metric-card {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.08);
    }
    .doc-card {
        padding: 1rem;
        border-radius: 10px;
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.08);
        margin-bottom: 0.8rem;
    }
    .pill {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #e2f0ff;
        color: #0f4c81;
        font-size: 0.82rem;
        margin-right: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings, _graph = get_runtime()

st.markdown(
    """
    <div class="hero">
      <h1>Aviation Document Retrieval Demo</h1>
      <p>LangGraph orchestration with intent-aware routing, FAISS retrieval, grounded answer generation, and artifact tracing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Run Controls")
    selected_query = st.selectbox("Sample query", SAMPLE_QUERIES, index=0)
    query = st.text_area("User query", value=selected_query, height=120)
    strategy = st.selectbox("Retrieval strategy", ["hybrid", "semantic", "bm25", "metadata_first"], index=0)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=min(settings.default_top_k, 10))
    allow_local_fallback = st.toggle("Allow local fallback answer", value=True)
    run = st.button("Run Pipeline", type="primary", use_container_width=True)

    st.caption(f"Dataset: `{settings.data_path.name}`")
    st.caption(f"Intent mode: `{settings.input_intent_mode}`")
    st.caption(f"LangSmith tracing: `{settings.langsmith_tracing}`")

if run:
    with st.spinner("Running LangGraph workflow..."):
        result = run_query(query, top_k, strategy, allow_local_fallback)
    st.session_state["last_result"] = result

result = st.session_state.get("last_result")

if result:
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Intent", result.get("intent", "unknown"))
    metric_2.metric("Intent Source", result.get("intent_source", "unknown"))
    metric_3.metric("Top Docs", len(result.get("topk_docs", []) or []))
    risk = result.get("hallucination_risk")
    metric_4.metric("Hallucination Risk", f"{float(risk):.3f}" if risk is not None else "n/a")

    tab_answer, tab_docs, tab_routing, tab_artifacts = st.tabs(
        ["Grounded Answer", "Retrieved Docs", "Routing Trace", "Artifacts"]
    )

    with tab_answer:
        st.subheader("Answer")
        st.write(result.get("answer", ""))
        citations = result.get("citations", []) or []
        if citations:
            st.subheader("Citations")
            for citation in citations:
                doc_id = citation.get("doc_id", "unknown")
                reason = citation.get("reason", "")
                st.markdown(
                    f"<span class='pill'>{doc_id}</span> {reason}",
                    unsafe_allow_html=True,
                )

    with tab_docs:
        docs = result.get("topk_docs", []) or []
        if not docs:
            st.info("No retrieved documents available for this run.")
        for index, doc in enumerate(docs, start=1):
            metadata = doc.get("metadata", {}) or {}
            scores = doc.get("scores", {}) or {}
            st.markdown(
                f"""
                <div class="doc-card">
                  <strong>{index}. {doc.get("doc_id", "unknown")}</strong><br/>
                  <span class="pill">{metadata.get("document_type", "unknown")}</span>
                  <span class="pill">{metadata.get("airport", "") or "airport:n/a"}</span>
                  <p>{doc.get("chunk_text", "")}</p>
                  <code>{json.dumps(scores, ensure_ascii=False)}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_routing:
        st.subheader("Diagnostics")
        st.json(result.get("retrieval_diagnostics", {}) or {})
        st.subheader("Grounding Report")
        st.json(result.get("grounding_report", {}) or {})

    with tab_artifacts:
        st.write(f"Phase 1: `{settings.phase1_output_path}`")
        st.write(f"Phase 2: `{settings.phase2_output_path}`")
        st.write(f"Phase 3: `{settings.phase3_output_path}`")
else:
    st.info("Enter a query and run the pipeline to inspect routing, retrieval, and grounded output.")
