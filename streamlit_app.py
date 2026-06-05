from __future__ import annotations

import base64
import json
import os
from dataclasses import replace
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:
    px = None
    go = None

from aviation_rag.config import Settings, configure_tracing_env, ensure_artifact_dirs
from aviation_rag.graph import build_graph
from aviation_rag.runtime import build_run_state
from aviation_rag.streamlit_bootstrap import ensure_streamlit_runtime

ensure_streamlit_runtime(module_name=__name__, script_path=__file__)

SAMPLE_QUERIES = [
    "engine failure after takeoff with emergency return",
    "den bao ENG OIL PRESS sang thi lam gi?",
    "crosswind turbulence during final approach at runway 25",
    "what is the meaning of MEL in aviation?",
]

STRATEGIES = ["hybrid", "semantic", "bm25", "metadata_first", "hybrid_rrf"]
RUN_MODES = ["Fast local", "Full dense/OpenAI"]
BACKGROUND_IMAGE_PATH = Path(__file__).parent / "assets" / "aviation-safety-bg.png"

SOURCE_NOTES = [
    "RAG is useful here because aviation answers should be grounded in retrieved safety reports, not guessed from model memory.",
    "Hybrid retrieval matters because ASRS reports contain both exact terms such as MEL and narrative language about incidents.",
    "The UI exposes retrieval scores and citations so a reviewer can inspect why an answer was produced.",
]


def asset_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "png"
    return f"data:image/{suffix};base64,{encoded}"


def render_html(html: str) -> None:
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def sync_query_from_preset() -> None:
    st.session_state["safety_query"] = st.session_state.get("scenario_preset", SAMPLE_QUERIES[0])


@st.cache_resource(show_spinner=False)
def get_runtime(intent_mode: str, run_mode: str) -> tuple[Settings, object]:
    base_settings = Settings()
    streamlit_tracing = os.getenv("STREAMLIT_LANGSMITH_TRACING", "false")
    if run_mode == "Fast local":
        settings = replace(
            base_settings,
            input_intent_mode=intent_mode,
            langsmith_tracing=streamlit_tracing,
            phase2_embedding_model="tfidf_svd_fallback",
            phase2_index_dir=base_settings.artifacts_dir / "phase2_index_fast",
            retrieval_max_docs=min(base_settings.retrieval_max_docs, 6000),
            retrieval_svd_components=min(base_settings.retrieval_svd_components, 96),
        )
    else:
        settings = replace(
            base_settings,
            input_intent_mode=intent_mode,
            langsmith_tracing=streamlit_tracing,
        )
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
    run_mode: str,
    write_artifacts: bool,
) -> dict[str, Any]:
    settings, graph = get_runtime(intent_mode, run_mode)
    state = build_run_state(
        settings,
        query_raw=query,
        top_k=top_k,
        strategy=strategy,
        allow_local_fallback=allow_local_fallback,
        force_local_answer=run_mode == "Fast local",
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


def compact_value(value: Any, max_chars: int = 32) -> str:
    text = str(value if value is not None else "n/a")
    if len(text) <= max_chars:
        return text
    head = max(8, max_chars // 2)
    tail = max(6, max_chars - head - 3)
    return f"{text[:head]}...{text[-tail:]}"


def render_card_grid(items: list[dict[str, Any]], css_class: str = "diag-grid") -> None:
    cards = []
    for item in items:
        label = escape(str(item.get("label", "")))
        value = str(item.get("value", "n/a"))
        display = escape(compact_value(value, int(item.get("max_chars", 32))))
        title = escape(value, quote=True)
        note = str(item.get("note", "") or "")
        note_html = f"<em>{escape(note)}</em>" if note else ""
        cards.append(
            f'<div class="diag-card"><span>{label}</span>'
            f'<strong title="{title}">{display}</strong>{note_html}</div>'
        )
    safe_class = escape(css_class, quote=True)
    render_html(f'<div class="{safe_class}">{"".join(cards)}</div>')


def evidence_frame(docs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, doc in enumerate(docs, start=1):
        scores = doc.get("scores", {}) or {}
        metadata = doc.get("metadata", {}) or {}
        text = str(doc.get("chunk_text", ""))
        rows.append(
            {
                "Rank": index,
                "Doc ID": str(doc.get("doc_id", "unknown")),
                "Type": str(metadata.get("document_type", "unknown")),
                "Airport": str(metadata.get("airport", "") or "n/a"),
                "Semantic": float(scores.get("semantic", 0.0) or 0.0),
                "BM25": float(scores.get("bm25", 0.0) or 0.0),
                "Metadata": float(scores.get("metadata", 0.0) or 0.0),
                "Final": float(scores.get("final", 0.0) or 0.0),
                "Excerpt": text[:220] + ("..." if len(text) > 220 else ""),
            }
        )
    return pd.DataFrame(rows)


def render_3d_evidence_map(docs: list[dict[str, Any]]) -> None:
    frame = evidence_frame(docs)
    if frame.empty:
        st.info("Run the pipeline first to build a 3D retrieval map.")
        return
    if px is None or go is None:
        st.warning("Install Plotly to view the 3D retrieval map: `pip install plotly`.")
        st.dataframe(frame, width="stretch", hide_index=True)
        return

    frame["Marker Size"] = (frame["Final"].clip(lower=0.02) * 36).clip(lower=8, upper=26)
    fig = px.scatter_3d(
        frame,
        x="Semantic",
        y="BM25",
        z="Metadata",
        color="Final",
        size="Marker Size",
        text="Doc ID",
        hover_name="Doc ID",
        hover_data={
            "Rank": True,
            "Type": True,
            "Airport": True,
            "Excerpt": True,
            "Marker Size": False,
        },
        color_continuous_scale=["#2f5d62", "#d95d39", "#f4d35e"],
        height=560,
    )
    fig.update_traces(marker={"opacity": 0.88, "line": {"width": 1, "color": "#132b2f"}})
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        scene={
            "xaxis_title": "Semantic score",
            "yaxis_title": "BM25 keyword score",
            "zaxis_title": "Metadata score",
            "camera": {"eye": {"x": 1.45, "y": 1.55, "z": 1.05}},
        },
        coloraxis_colorbar={"title": "Final"},
    )
    st.plotly_chart(fig, width="stretch")

    methods = ["Semantic", "BM25", "Metadata", "Final"]
    surface = frame[methods].to_numpy().transpose()
    terrain = go.Figure(
        data=[
            go.Surface(
                z=surface,
                x=frame["Rank"].tolist(),
                y=methods,
                colorscale=[[0, "#2f5d62"], [0.55, "#d95d39"], [1, "#f4d35e"]],
                colorbar={"title": "Score"},
            )
        ]
    )
    terrain.update_layout(
        height=420,
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        scene={
            "xaxis_title": "Document rank",
            "yaxis_title": "Scoring signal",
            "zaxis_title": "Score",
            "camera": {"eye": {"x": 1.4, "y": -1.55, "z": 0.95}},
        },
    )
    st.plotly_chart(terrain, width="stretch")


def method_badges(diagnostics: dict[str, Any]) -> None:
    latency = diagnostics.get("latency_ms")
    render_card_grid(
        [
            {"label": "Backend", "value": diagnostics.get("retrieval_backend", "unknown"), "note": "retrieval engine"},
            {"label": "Embedding", "value": diagnostics.get("embedding_backend", "unknown"), "note": "vector backend"},
            {"label": "Dim", "value": diagnostics.get("embedding_dim", "n/a"), "note": "vector size"},
            {"label": "Fusion", "value": diagnostics.get("fusion_method", "n/a"), "note": "score combiner"},
            {
                "label": "Latency",
                "value": f"{float(latency):.1f} ms" if latency is not None else "n/a",
                "note": "runtime",
            },
        ]
    )


def show_warnings(diagnostics: dict[str, Any], run_mode: str) -> None:
    adapter_mode = diagnostics.get("adapter_mode")
    embedding_backend = diagnostics.get("embedding_backend")
    fallback_reason = diagnostics.get("fallback_reason")
    if adapter_mode == "generated_mock":
        st.warning("Phase 2 is using generated mock retrieval, not real corpus search.")
    if embedding_backend == "tfidf_svd_faiss_fallback":
        if run_mode == "Fast local":
            render_html(
                '<div class="notice-card notice-info">'
                "<strong>Fast local mode:</strong> using TF-IDF/SVD + FAISS fallback so the demo returns quickly "
                "without downloading MiniLM. Switch to Full dense/OpenAI when you want the full embedding pipeline."
                "</div>"
            )
        else:
            st.warning("Dense MiniLM embeddings were unavailable; semantic search is using explicit TF-IDF/SVD FAISS fallback.")
    if embedding_backend == "unavailable":
        st.error(f"Phase 2 retrieval backend is unavailable: {fallback_reason}")


def render_answer_card(answer: Any) -> None:
    text = str(answer or "No grounded answer was produced.")
    html = escape(text).replace("\n", "<br/>")
    render_html(f'<div class="answer-card">{html}</div>')


def render_citation_cards(citations: list[dict[str, Any]]) -> None:
    if not citations:
        st.info("No citations were returned. Check the Evidence tab for retrieved reports.")
        return
    rows = []
    for citation in citations:
        doc_id = escape(str(citation.get("doc_id", "unknown")))
        reason = escape(str(citation.get("reason", "Cited evidence")))
        rows.append(
            f'<div class="citation-card"><span>{doc_id}</span><strong>{reason}</strong></div>'
        )
    render_html(f'<div class="citation-grid">{"".join(rows)}</div>')


def render_phase_cards() -> None:
    render_html(
        '<div class="phase-grid">'
        '<div class="phase-card"><span>Phase 1</span><strong>Intent routing</strong><em>Understand query type</em></div>'
        '<div class="phase-card"><span>Phase 2</span><strong>Hybrid retrieval</strong><em>Find ranked ASRS evidence</em></div>'
        '<div class="phase-card"><span>Phase 3</span><strong>Grounded answer</strong><em>Answer with citations</em></div>'
        "</div>"
    )


def render_run_result(result: dict[str, Any], run_mode: str) -> None:
    diagnostics = result.get("retrieval_diagnostics", {}) or {}
    top_docs = result.get("topk_docs", []) or []
    risk = result.get("hallucination_risk")
    st.markdown("### Run result")
    show_warnings(diagnostics, run_mode)
    render_card_grid(
        [
            {"label": "Intent", "value": result.get("intent", "unknown"), "note": "Phase 1 route"},
            {"label": "Intent source", "value": result.get("intent_source", "unknown"), "note": "heuristic / ML"},
            {"label": "Top docs", "value": len(top_docs), "note": "retrieved evidence"},
            {
                "label": "Risk proxy",
                "value": f"{float(risk):.3f}" if risk is not None else "n/a",
                "note": "lower is better",
            },
        ],
        css_class="result-grid",
    )
    method_badges(diagnostics)
    st.markdown("#### Grounded answer")
    render_answer_card(result.get("answer", ""))
    st.markdown("#### Citations")
    render_citation_cards(result.get("citations", []) or [])
    with st.expander("Technical diagnostics JSON", expanded=False):
        st.json(diagnostics)


st.set_page_config(
    page_title="Aviation Safety Intelligence",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

background_data_uri = asset_data_uri(BACKGROUND_IMAGE_PATH)

st.markdown(
    f"""
    <style>
    :root {{
        --ink: #101820;
        --muted: #51606a;
        --panel: rgba(255, 255, 255, 0.9);
        --line: rgba(16, 24, 32, 0.12);
        --teal: #12343b;
        --copper: #c7522a;
        --amber: #f2c14e;
    }}
    .stApp {{
        background: #e9eef0;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: -14px;
        background:
            linear-gradient(90deg, rgba(8, 22, 27, 0.72) 0%, rgba(8, 22, 27, 0.45) 42%, rgba(247, 248, 251, 0.78) 100%),
            url("{background_data_uri}");
        background-size: cover;
        background-position: center;
        filter: blur(3px);
        transform: scale(1.02);
        z-index: -2;
    }}
    .stApp::after {{
        content: "";
        position: fixed;
        inset: 0;
        background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(244,247,248,0.82) 58%, rgba(247,248,251,0.96) 100%);
        z-index: -1;
    }}
    header[data-testid="stHeader"] {{
        background: transparent;
        height: 2.75rem;
        z-index: 999;
    }}
    [data-testid="stToolbar"] {{
        visibility: visible;
        height: auto;
    }}
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 1001 !important;
        left: 0.65rem !important;
        top: 0.65rem !important;
    }}
    .block-container {{
        padding-top: 0.25rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }}
    [data-testid="stSidebar"] {{
        background:
            linear-gradient(180deg, rgba(245, 248, 250, 0.96), rgba(235, 242, 244, 0.92));
        border-right: 1px solid rgba(16, 24, 32, 0.08);
        backdrop-filter: blur(16px);
    }}
    .status-strip {{
        display: grid;
        grid-template-columns: 1fr 1.2fr 1.6fr;
        gap: 0.55rem;
        margin: 0.7rem 0 0.65rem 0;
    }}
    .status-chip {{
        padding: 0.55rem 0.7rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.86);
        border: 1px solid var(--line);
        color: #34434b;
        font-size: 0.84rem;
        overflow-wrap: anywhere;
    }}
    .status-chip strong {{
        display: block;
        color: var(--teal);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.15rem;
    }}
    .hero {{
        min-height: 254px;
        padding: 1.2rem 1.25rem;
        border-radius: 8px;
        color: white;
        background:
            linear-gradient(90deg, rgba(9, 28, 34, 0.92) 0%, rgba(18, 52, 59, 0.78) 46%, rgba(18, 52, 59, 0.2) 100%),
            url("{background_data_uri}");
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: 0 22px 70px rgba(9, 28, 34, 0.28);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .eyebrow {{
        display: inline-block;
        width: fit-content;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: rgba(242, 193, 78, 0.16);
        border: 1px solid rgba(242, 193, 78, 0.34);
        color: #ffe7a3;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .hero h1 {{
        margin: 0.82rem 0 0 0;
        max-width: 760px;
        font-size: 2.72rem;
        line-height: 1.04;
        letter-spacing: 0;
    }}
    .hero p {{
        margin: 0.7rem 0 0 0;
        max-width: 730px;
        color: #edf7f4;
        font-size: 1rem;
    }}
    .hero-strip {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.5rem;
        margin-top: 1rem;
        max-width: 840px;
    }}
    .hero-stat {{
        padding: 0.58rem 0.68rem;
        border-radius: 8px;
        background: rgba(6, 22, 27, 0.46);
        border: 1px solid rgba(255,255,255,0.14);
        backdrop-filter: blur(10px);
    }}
    .hero-stat strong {{
        display: block;
        color: white;
        font-size: 0.95rem;
    }}
    .hero-stat span {{
        color: #d7e6e4;
        font-size: 0.78rem;
    }}
    .workflow-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }}
    .workflow-step {{
        min-height: 8.5rem;
        padding: 0.9rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.9);
        border: 1px solid var(--line);
        box-shadow: 0 12px 32px rgba(16, 24, 32, 0.08);
    }}
    .workflow-step b {{
        display: inline-flex;
        width: 1.7rem;
        height: 1.7rem;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: #12343b;
        color: white;
        margin-bottom: 0.65rem;
    }}
    .workflow-step strong {{
        display: block;
        color: var(--teal);
        margin-bottom: 0.3rem;
    }}
    .workflow-step span {{
        color: #42515a;
        font-size: 0.92rem;
    }}
    .console-panel {{
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        box-shadow: 0 18px 48px rgba(16, 24, 32, 0.1);
    }}
    .section-panel {{
        padding: 1rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.88);
        border: 1px solid var(--line);
        margin-bottom: 0.85rem;
    }}
    .mode-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.8rem 0 1rem 0;
    }}
    .mode-card {{
        padding: 0.85rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.9);
        border: 1px solid var(--line);
    }}
    .mode-card strong {{
        display: block;
        color: var(--teal);
        margin-bottom: 0.25rem;
    }}
    .mode-card span {{
        color: #42515a;
        font-size: 0.9rem;
    }}
    .mission-grid {{
        display: grid;
        grid-template-columns: 1.1fr 0.9fr;
        gap: 0.8rem;
        margin-top: 0.8rem;
    }}
    .mission-card {{
        padding: 1rem;
        border-radius: 8px;
        background: var(--panel);
        border: 1px solid var(--line);
        min-height: 7.4rem;
        box-shadow: 0 10px 30px rgba(16, 24, 32, 0.08);
    }}
    .mission-card strong {{
        display: block;
        color: var(--teal);
        margin-bottom: 0.35rem;
    }}
    .mission-card span {{
        color: #42515a;
        font-size: 0.94rem;
    }}
    .decision-note {{
        margin-top: 0.8rem;
        padding: 0.9rem 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--copper);
        background: rgba(255,255,255,0.9);
        color: #24323a;
    }}
    .pill {{
        display: inline-block;
        padding: 0.22rem 0.58rem;
        border-radius: 999px;
        background: #e6efe9;
        color: #173b2f;
        font-size: 0.82rem;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
    }}
    .doc-card {{
        padding: 1rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(15, 23, 42, 0.08);
        margin-bottom: 0.8rem;
    }}
    .result-grid, .diag-grid, .phase-grid, .citation-grid {{
        display: grid;
        gap: 0.7rem;
        margin: 0.75rem 0 1rem 0;
    }}
    .result-grid {{
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .diag-grid {{
        grid-template-columns: repeat(5, minmax(0, 1fr));
    }}
    .phase-grid {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .citation-grid {{
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .diag-card, .phase-card, .citation-card {{
        min-width: 0;
        padding: 0.78rem 0.85rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        box-shadow: 0 10px 28px rgba(16, 24, 32, 0.07);
    }}
    .diag-card span, .phase-card span {{
        display: block;
        color: #5b6870;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.28rem;
    }}
    .diag-card strong, .phase-card strong {{
        display: block;
        color: var(--ink);
        font-size: clamp(1rem, 1.5vw, 1.28rem);
        line-height: 1.18;
        overflow-wrap: anywhere;
    }}
    .diag-card em, .phase-card em {{
        display: block;
        margin-top: 0.32rem;
        color: #687780;
        font-style: normal;
        font-size: 0.78rem;
    }}
    .phase-card {{
        background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(235,244,241,0.9));
    }}
    .answer-card {{
        padding: 1rem 1.05rem;
        border-radius: 8px;
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(16, 24, 32, 0.1);
        border-left: 4px solid var(--teal);
        color: #20313a;
        line-height: 1.68;
        box-shadow: 0 14px 36px rgba(16, 24, 32, 0.08);
        overflow-wrap: anywhere;
    }}
    .citation-card span {{
        display: inline-block;
        padding: 0.16rem 0.5rem;
        border-radius: 999px;
        background: #e6efe9;
        color: #173b2f;
        font-size: 0.8rem;
        margin-bottom: 0.35rem;
    }}
    .citation-card strong {{
        display: block;
        color: #25343c;
        font-size: 0.92rem;
        font-weight: 600;
    }}
    .notice-card {{
        margin: 0.65rem 0 0.85rem 0;
        padding: 0.8rem 0.95rem;
        border-radius: 8px;
        border-left: 4px solid var(--teal);
        background: rgba(237, 246, 244, 0.95);
        color: #263b42;
        line-height: 1.5;
    }}
    .notice-card strong {{
        color: var(--teal);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.2rem;
        border-bottom: 1px solid rgba(16, 24, 32, 0.12);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px 6px 0 0;
        padding: 0.65rem 0.85rem;
    }}
    @media (max-width: 900px) {{
        .mission-grid, .hero-strip, .workflow-grid, .mode-grid, .status-strip,
        .result-grid, .diag-grid, .phase-grid {{ grid-template-columns: 1fr; }}
        .hero h1 {{ font-size: 2rem; }}
        .hero {{ min-height: auto; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div>
        <span class="eyebrow">ASRS Incident Intelligence</span>
        <h1>Aviation Safety Query Console</h1>
        <p>Investigate aviation hazards through ASRS evidence, retrieval diagnostics, and grounded answer review.</p>
      </div>
      <div class="hero-strip">
        <div class="hero-stat"><strong>Query</strong><span>hazard, incident, term</span></div>
        <div class="hero-stat"><strong>Evidence</strong><span>ranked ASRS reports</span></div>
        <div class="hero-stat"><strong>Grounding</strong><span>citations and risk proxy</span></div>
        <div class="hero-stat"><strong>Compare</strong><span>BM25, FAISS, hybrid, RRF</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

settings_preview = Settings()
st.markdown(
    f"""
    <div class="status-strip">
      <div class="status-chip"><strong>Dataset</strong>{settings_preview.data_path.name}</div>
      <div class="status-chip"><strong>Embedding</strong>{settings_preview.phase2_embedding_model}</div>
      <div class="status-chip"><strong>Index</strong>{settings_preview.phase2_index_dir}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

result = st.session_state.get("last_result")
last_config = st.session_state.get(
    "last_config",
    {
        "query": SAMPLE_QUERIES[0],
        "top_k": 5,
        "strategy": "hybrid",
        "intent_mode": "heuristic",
        "run_mode": "Fast local",
        "allow_local_fallback": True,
        "write_artifacts": True,
    },
)
if "safety_query" not in st.session_state:
    st.session_state["safety_query"] = str(last_config.get("query") or SAMPLE_QUERIES[0])
if "scenario_preset" not in st.session_state:
    current_query = str(st.session_state["safety_query"])
    st.session_state["scenario_preset"] = current_query if current_query in SAMPLE_QUERIES else SAMPLE_QUERIES[0]

tab_analyze, tab_evidence_root, tab_research = st.tabs(["Analyze", "Evidence", "Research"])
tab_overview = tab_run = tab_analyze
tab_insights = tab_compare = tab_evidence = tab_evidence_root

with tab_overview:
    st.subheader("Analyze safety query")
    st.write(
        "Main workflow: enter a hazard, incident, aviation term, or maintenance concern; "
        "then inspect the grounded answer and evidence."
    )
    preset_col, mode_col = st.columns([1.45, 0.85])
    with preset_col:
        preset = st.selectbox(
            "Scenario preset",
            SAMPLE_QUERIES,
            key="scenario_preset",
            on_change=sync_query_from_preset,
        )
    with mode_col:
        run_mode = st.selectbox(
            "Run mode",
            RUN_MODES,
            index=RUN_MODES.index(str(last_config.get("run_mode", "Fast local"))),
        )
    query = st.text_area("Safety query", key="safety_query", height=86)
    control_1, control_2, control_3 = st.columns([1, 1, 1])
    with control_1:
        intent_values = ["heuristic", "auto", "ml"]
        intent_mode = st.selectbox(
            "Intent mode",
            intent_values,
            index=intent_values.index(str(last_config.get("intent_mode", "heuristic"))),
        )
    with control_2:
        strategy = st.selectbox(
            "Retrieval strategy",
            STRATEGIES,
            index=STRATEGIES.index(str(last_config.get("strategy", "hybrid"))),
        )
    with control_3:
        top_k = st.slider("Top K", min_value=1, max_value=10, value=int(last_config.get("top_k", 5)))
    c_toggle_1, c_toggle_2 = st.columns(2)
    with c_toggle_1:
        allow_local_fallback = c_toggle_1.toggle(
            "Local fallback",
            value=bool(last_config.get("allow_local_fallback", True)),
        )
    with c_toggle_2:
        write_artifacts = c_toggle_2.toggle("Artifacts", value=bool(last_config.get("write_artifacts", True)))
    run = st.button("Analyze Safety Query", type="primary", width="stretch")
    current_config = {
        "query": query,
        "top_k": top_k,
        "strategy": strategy,
        "intent_mode": intent_mode,
        "run_mode": run_mode,
        "allow_local_fallback": allow_local_fallback,
        "write_artifacts": write_artifacts,
    }
    if run:
        st.session_state["last_config"] = current_config
        with st.spinner("Running intent routing, retrieval, and grounded answer generation..."):
            st.session_state["last_result"] = run_query(
                query=query,
                top_k=top_k,
                strategy=strategy,
                allow_local_fallback=allow_local_fallback,
                intent_mode=intent_mode,
                run_mode=run_mode,
                write_artifacts=write_artifacts,
            )
        result = st.session_state.get("last_result")

    saved_config = st.session_state.get("last_config", {}) or {}
    result_is_stale = bool(result) and any(
        current_config.get(key) != saved_config.get(key)
        for key in ["query", "top_k", "strategy", "intent_mode", "run_mode", "allow_local_fallback"]
    )
    if result:
        if result_is_stale:
            st.info("Inputs changed. Click Analyze Safety Query to refresh the answer and evidence.")
        else:
            result_run_mode = str(saved_config.get("run_mode", run_mode))
            render_run_result(result, result_run_mode)
    else:
        st.info("Run the query above to inspect answer, routing, retrieval, and artifacts.")

    st.subheader("What the user query is for")
    st.write(
        "A user enters a query to investigate an aviation safety situation against the ASRS incident-report corpus. "
        "The app retrieves similar reports, shows the evidence behind the ranking, and produces a grounded research answer that can be checked through citations."
    )
    st.markdown(
        """
        <div class="workflow-grid">
          <div class="workflow-step">
            <b>1</b>
            <strong>Describe the situation</strong>
            <span>Enter an incident, hazard, aviation term, airport condition, or maintenance concern.</span>
          </div>
          <div class="workflow-step">
            <b>2</b>
            <strong>Route the intent</strong>
            <span>Classify the query as incident, procedure, metadata, or factoid so retrieval uses the right strategy.</span>
          </div>
          <div class="workflow-step">
            <b>3</b>
            <strong>Retrieve evidence</strong>
            <span>Search ASRS narratives with semantic vectors, BM25 keywords, metadata signals, or hybrid fusion.</span>
          </div>
          <div class="workflow-step">
            <b>4</b>
            <strong>Review the answer</strong>
            <span>Inspect citations, scores, diagnostics, and hallucination-risk proxy before trusting the output.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mission-grid">
          <div class="mission-card">
            <strong>Best fit</strong>
            <span>Tra cứu báo cáo sự cố tương tự, phân tích hazard pattern, tìm context cho incident review, và so sánh thuật toán retrieval trong project RAG.</span>
          </div>
          <div class="mission-card">
            <strong>Not a replacement for</strong>
            <span>QRH, SOP, aircraft manual, maintenance release authority, ATC instruction, or official emergency decision-making.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="decision-note">
        <strong>Final purpose:</strong> this is an aviation safety incident intelligence dashboard. It is for research, evidence discovery, and risk-context analysis. It is not an official aircraft emergency checklist or a replacement for SOP/QRH/maintenance authority.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_phase_cards()
    st.markdown(
        """
        <div class="decision-note">
        <strong>Current limitation:</strong> the ASRS dataset is strong for incident narratives. For clean definitions such as MEL, the next practical upgrade is adding FAA glossary/manual/procedure documents to the retrieval corpus.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_insights:
    st.subheader("3D evidence map")
    st.write(
        "Each point is a retrieved ASRS document. The 3D position shows how semantic similarity, keyword matching, and metadata signals combine into the final rank."
    )
    if not result:
        st.info("Run a query in Analyze to generate the retrieval map.")
    else:
        render_3d_evidence_map(result.get("topk_docs", []) or [])

with tab_compare:
    st.subheader("Compare retrieval strategies")
    st.write("Run the same query through BM25, semantic FAISS, weighted hybrid, metadata-first, and RRF hybrid.")
    compare = st.button("Compare all methods", width="stretch")
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
                    run_mode=run_mode,
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
            doc_id = escape(str(doc.get("doc_id", "unknown")))
            doc_type = escape(str(metadata.get("document_type", "unknown")))
            airport = escape(str(metadata.get("airport", "") or "n/a"))
            chunk_text = escape(str(doc.get("chunk_text", "")))
            score_json = escape(json.dumps(scores, ensure_ascii=False))
            st.markdown(
                f"""
                <div class="doc-card">
                  <strong>{index}. {doc_id}</strong><br/>
                  <span class="pill">{doc_type}</span>
                  <span class="pill">airport: {airport}</span>
                  <span class="pill">final: {float(scores.get('final', 0.0)):.4f}</span>
                  <p>{chunk_text}</p>
                  <code>{score_json}</code>
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
    st.subheader("Core formulas")
    st.latex(r"cos(q, d)=\frac{q \cdot d}{\|q\|\|d\|}")
    st.latex(r"BM25(q,d)=\sum_{t \in q} IDF(t)\frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b\frac{|d|}{avgdl})}")
    st.latex(r"score_{hybrid}=0.50\,score_{semantic}+0.35\,score_{BM25}+0.15\,score_{metadata}")
    st.latex(r"RRF(d)=\sum_{r \in R}\frac{1}{k + rank_r(d)}")
    st.latex(r"MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}")
    st.subheader("Research direction")
    for note in SOURCE_NOTES:
        st.info(note)
    st.markdown(
        """
        - Hybrid retrieval: compare sparse BM25, dense embeddings, weighted fusion, and RRF for each aviation intent.
        - Groundedness: evaluate whether each major answer claim is supported by retrieved citations.
        - UI explainability: keep score decomposition, diagnostics, and 3D evidence maps visible for review.
        - Next upgrade: add reranking and a small labeled evaluation set with Recall@K, MRR, answer faithfulness, and citation support.
        """
    )
    st.subheader("Artifact paths")
    active_config = st.session_state.get("last_config", last_config)
    settings, _graph = get_runtime(
        str(active_config.get("intent_mode", "heuristic")),
        str(active_config.get("run_mode", "Fast local")),
    )
    st.write(f"Phase 1: `{settings.phase1_output_path}`")
    st.write(f"Phase 2: `{settings.phase2_output_path}`")
    st.write(f"Phase 2 index: `{settings.phase2_index_dir}`")
    st.write(f"Phase 3: `{settings.phase3_output_path}`")
