"""
Aviation Safety Intelligence — Streamlit Demo UI (v3)
======================================================
Redesigned with focus on:
  - Grounded Answer card (top of results)
  - Evidence-first layout (not ML dashboard)
  - Minimal technical jargon on surface
  - Clean, trustworthy aviation feel

Usage:
    streamlit run app.py

Authors: Vinh Hoang & Quan San
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aviation_rag.config import Settings
from aviation_rag.phase1_hoang_intent_routing import Phase1HoangIntentRouting
from aviation_rag.phase2_san_contract_adapter import Phase2SanContractAdapter
from aviation_rag.schemas import InputAgentOutput, MiddleAgentOutput

# ═════════════════════════════════════════════════════════════════════
# Page Config
# ═════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Aviation Safety Intelligence",
    page_icon="https://em-content.zobj.net/source/twitter/408/airplane_2708-fe0f.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═════════════════════════════════════════════════════════════════════
# CSS
# ═════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --navy-900: #0c1829;
    --navy-800: #122240;
    --navy-700: #1a3158;
    --teal-700: #0f766e;
    --teal-600: #0d9488;
    --teal-500: #14b8a6;
    --teal-100: #ccfbf1;
    --teal-50: #f0fdfa;
    --sky-500: #0ea5e9;
    --sky-100: #e0f2fe;
    --amber-500: #f59e0b;
    --amber-100: #fef3c7;
    --rose-500: #f43f5e;
    --rose-100: #ffe4e6;
    --emerald-500: #10b981;
    --emerald-100: #d1fae5;
    --slate-900: #0f172a;
    --slate-700: #334155;
    --slate-600: #475569;
    --slate-500: #64748b;
    --slate-400: #94a3b8;
    --slate-300: #cbd5e1;
    --slate-200: #e2e8f0;
    --slate-100: #f1f5f9;
    --slate-50: #f8fafc;
    --white: #ffffff;
}

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--slate-50);
}
section[data-testid="stSidebar"] {
    background: var(--white);
    border-right: 1px solid var(--slate-200);
}

/* ── Top Bar ─────────────────────────────────── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--slate-200);
    margin-bottom: 1.5rem;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.topbar-icon {
    font-size: 1.3rem;
}
.topbar-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--navy-800);
    letter-spacing: -0.01em;
}
.topbar-badge {
    font-size: 0.7rem;
    color: var(--slate-400);
    background: var(--slate-100);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-weight: 500;
}

/* ── Search Hero ─────────────────────────────── */
.search-hero {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
}
.search-hero h1 {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--navy-800);
    margin: 0;
    letter-spacing: -0.03em;
}
.search-hero-sub {
    font-size: 0.95rem;
    color: var(--slate-500);
    margin-top: 0.35rem;
    font-weight: 400;
}
.search-hero-powered {
    font-size: 0.78rem;
    color: var(--slate-400);
    margin-top: 0.5rem;
}
.search-hero-powered strong {
    color: var(--teal-600);
    font-weight: 600;
}

/* ── Scenario Chips ──────────────────────────── */
.chips-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.75rem 0 1rem;
}

/* ── Pipeline Status ─────────────────────────── */
.pipeline-status {
    text-align: center;
    padding: 0.6rem 1rem;
    margin: 0.5rem auto 1.25rem;
    max-width: 720px;
    font-size: 0.82rem;
    color: var(--slate-500);
    background: var(--white);
    border: 1px solid var(--slate-200);
    border-radius: 8px;
}
.pipeline-status strong {
    color: var(--navy-800);
    font-weight: 600;
}
.pipeline-status .teal {
    color: var(--teal-600);
    font-weight: 700;
}

/* ── Answer Card ─────────────────────────────── */
.answer-card {
    background: var(--white);
    border: 1px solid var(--teal-100);
    border-left: 4px solid var(--teal-500);
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(13,148,136,0.06);
}
.answer-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}
.answer-card-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    color: var(--teal-600);
}
.answer-card-body {
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--slate-700);
}
.answer-card-footer {
    margin-top: 0.75rem;
    font-size: 0.75rem;
    color: var(--slate-400);
    font-style: italic;
}

/* ── Evidence Section ────────────────────────── */
.evidence-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}
.evidence-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--navy-800);
}
.evidence-count {
    font-size: 0.78rem;
    color: var(--slate-400);
    font-weight: 500;
}

/* ── Evidence Card ───────────────────────────── */
.ev-card {
    background: var(--white);
    border: 1px solid var(--slate-200);
    border-radius: 10px;
    padding: 1.1rem 1.35rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.15s;
}
.ev-card:hover {
    border-color: var(--teal-500);
}
.ev-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.ev-rank {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}
.rank-dot {
    width: 22px; height: 22px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--white);
}
.rd-1 { background: linear-gradient(135deg, #f59e0b, #ef4444); }
.rd-2 { background: linear-gradient(135deg, #8b5cf6, #6366f1); }
.rd-3 { background: linear-gradient(135deg, #0ea5e9, #3b82f6); }
.rd-n { background: var(--slate-400); }
.ev-id {
    font-size: 0.73rem;
    color: var(--slate-400);
    font-weight: 500;
}
.ev-match {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--teal-700);
    background: var(--teal-50);
    padding: 0.15rem 0.55rem;
    border-radius: 5px;
}
.ev-text {
    font-size: 0.86rem;
    line-height: 1.6;
    color: var(--slate-600);
    margin-bottom: 0.5rem;
}
.ev-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
}
.ev-chip {
    font-size: 0.68rem;
    font-weight: 500;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    background: var(--slate-100);
    color: var(--slate-500);
    border: 1px solid var(--slate-200);
}

/* ── Intent Pill ─────────────────────────────── */
.intent-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
}
.ip-incident { background: var(--rose-100); color: var(--rose-500); }
.ip-procedure { background: var(--sky-100); color: var(--sky-500); }
.ip-metadata { background: var(--amber-100); color: var(--amber-500); }
.ip-factoid { background: var(--emerald-100); color: var(--emerald-500); }

/* ── Empty State ─────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem 2rem;
}
.empty-icon { font-size: 2.5rem; opacity: 0.5; margin-bottom: 0.75rem; }
.empty-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--slate-500);
}
.empty-sub {
    font-size: 0.85rem;
    color: var(--slate-400);
    max-width: 380px;
    margin: 0.3rem auto 0;
    line-height: 1.5;
}

/* ── How It Works (empty state) ──────────────── */
.hiw-strip {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    gap: 0;
    padding: 1.5rem 0;
    max-width: 600px;
    margin: 0 auto;
}
.hiw-step {
    flex: 1;
    text-align: center;
    padding: 0 0.5rem;
}
.hiw-num {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--white);
    margin-bottom: 0.4rem;
}
.hiw-n1 { background: var(--sky-500); }
.hiw-n2 { background: var(--teal-500); }
.hiw-n3 { background: var(--amber-500); }
.hiw-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--slate-700);
}
.hiw-desc {
    font-size: 0.68rem;
    color: var(--slate-400);
    margin-top: 0.15rem;
}
.hiw-arrow {
    color: var(--slate-300);
    font-size: 1rem;
    padding-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# Init
# ═════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initializing retrieval engine...")
def load_pipeline():
    settings = Settings(langsmith_tracing="false", input_intent_mode="heuristic")
    return settings, Phase1HoangIntentRouting(settings), Phase2SanContractAdapter(settings)

settings, phase1, adapter = load_pipeline()


# ═════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════

INTENT_MAP = {
    "Incident_Report":     ("Incident Report",     "ip-incident"),
    "Technical_Procedure": ("Technical Procedure",  "ip-procedure"),
    "Metadata_Query":      ("Operational Context",  "ip-metadata"),
    "Factoid":             ("Quick Fact",           "ip-factoid"),
}

SCENARIOS = [
    "Engine failure after takeoff",
    "Runway incursion during taxi",
    "Bird strike on approach",
    "TCAS alert during climb",
    "Severe turbulence en route",
    "Hydraulic system failure",
]

META_FIELDS = [
    ("aircraft1_model",        "Aircraft"),
    ("aircraft1_flight_phase", "Phase"),
    ("primary_problem",        "Factor"),
    ("location_state",         "Location"),
    ("event_outcome",          "Outcome"),
]


def _synthesize_answer(docs, intent: str, query: str) -> str:
    """Build a simple grounded summary from top retrieved docs."""
    if not docs:
        return "No matching incident reports were found for this query."

    n = len(docs)
    # Extract key metadata patterns
    phases = set()
    aircraft = set()
    factors = set()
    for d in docs[:5]:
        m = d.metadata or {}
        fp = m.get("aircraft1_flight_phase")
        ac = m.get("aircraft1_model")
        pf = m.get("primary_problem")
        if fp and str(fp).lower() not in ("nan", "none", ""):
            phases.add(str(fp).strip())
        if ac and str(ac).lower() not in ("nan", "none", ""):
            aircraft.add(str(ac).strip())
        if pf and str(pf).lower() not in ("nan", "none", ""):
            factors.add(str(pf).strip())

    lines = []
    lines.append(f"Based on **{n} matching ASRS incident reports**, ")

    # Try to build a context-aware summary
    if phases:
        lines.append(f"these incidents most commonly occur during the **{', '.join(list(phases)[:3])}** phase(s) of flight. ")
    if aircraft:
        lines.append(f"Aircraft types involved include **{', '.join(list(aircraft)[:3])}**. ")
    if factors:
        lines.append(f"Primary contributing factors include **{', '.join(list(factors)[:3])}**. ")

    # Add a snippet from top result
    if docs:
        top_text = docs[0].chunk_text[:200].strip()
        lines.append(f'\n\nThe most relevant report states: *"{top_text}..."*')

    return "".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Sidebar (minimal)
# ═════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("#### Search Settings")
    top_k = st.slider("Number of results", 3, 15, 5, 1)
    search_mode = st.radio(
        "Search mode",
        ["Automatic", "Contextual", "Keyword", "Combined"],
        index=0,
        help="Automatic lets the system choose the best strategy based on your query.",
    )
    _mode_map = {"Automatic": None, "Contextual": "semantic", "Keyword": "bm25", "Combined": "hybrid"}
    strategy_override = _mode_map.get(search_mode)

    st.markdown("---")
    st.caption("Aviation Safety Intelligence  \nNLP Course Final Project  \nVinh Hoang & Quan San")


# ═════════════════════════════════════════════════════════════════════
# Top Bar
# ═════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <span class="topbar-icon">&#9992;&#65039;</span>
        <span class="topbar-title">Aviation Safety Intelligence</span>
    </div>
    <span class="topbar-badge">Powered by 111,000+ ASRS reports</span>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# Search Hero
# ═════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="search-hero">
    <h1>Search Aviation Incident Reports</h1>
    <p class="search-hero-sub">Get evidence-based answers from real safety occurrence data</p>
</div>
""", unsafe_allow_html=True)

# Search input
_, col_search, _ = st.columns([1.5, 4, 1.5])
with col_search:
    query = st.text_input(
        "Search",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g., engine failure after takeoff, runway incursion during taxi...",
        label_visibility="collapsed",
    )
    search_clicked = st.button("Search", type="primary", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# Scenario Chips
# ═════════════════════════════════════════════════════════════════════

st.markdown('<div style="text-align:center; margin-top:0.25rem;"><span style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; color:#94a3b8; font-weight:600;">Try a scenario</span></div>', unsafe_allow_html=True)

chip_row1 = st.columns(3)
chip_row2 = st.columns(3)
for i, label in enumerate(SCENARIOS):
    row = chip_row1 if i < 3 else chip_row2
    with row[i % 3]:
        if st.button(label, key=f"sc_{i}", use_container_width=True):
            st.session_state["query_input"] = label
            st.rerun()


# ═════════════════════════════════════════════════════════════════════
# RESULTS
# ═════════════════════════════════════════════════════════════════════

if search_clicked and query.strip():

    # ── Thinking Process (NotebookLM-style) ──────────────────────────
    t0 = time.time()

    with st.status("Analyzing your query...", expanded=True) as status:

        # Step 1: Query Analysis
        st.write("**Step 1 / 4** — Analyzing query")
        st.caption(f'Understanding: "{query.strip()}"')
        t1 = time.time()
        p1: InputAgentOutput = phase1.build_output(query.strip(), top_k=top_k, strategy=strategy_override)
        t1_ms = (time.time() - t1) * 1000

        intent_label, intent_css = INTENT_MAP.get(p1.intent, ("Unknown", "ip-incident"))
        st.markdown(f"Detected intent: **{intent_label}** (confidence: {p1.intent_confidence:.0%})")
        st.caption(f"Optimized query: \"{p1.rewritten_query}\"")
        if p1.expanded_queries:
            st.caption(f"Also searching for: {', '.join(p1.expanded_queries[:3])}")
        st.caption(f"Completed in {t1_ms:.0f}ms")

        # Step 2: Loading Index
        status.update(label="Loading search index...", expanded=True)
        st.write("**Step 2 / 4** — Preparing search engine")
        st.caption(f"Strategy: {p1.retrieval_plan.strategy} (fallback: {p1.retrieval_plan.fallback_strategy})")
        st.caption(f"Reason: {p1.retrieval_plan.routing_reason}")

        # Step 3: Retrieval
        status.update(label="Searching incident reports...", expanded=True)
        st.write("**Step 3 / 4** — Searching incident database")
        t3 = time.time()
        p2: MiddleAgentOutput = adapter.resolve_output(p1)
        t3_ms = (time.time() - t3) * 1000

        diag = p2.retrieval_diagnostics
        mode = diag.get("adapter_mode", "unknown")
        strategy_used = diag.get("strategy_used", p1.retrieval_plan.strategy)
        search_ms = diag.get("search_time_ms", t3_ms)
        index_size = diag.get("index_size", "N/A")
        index_str = f"{index_size:,}" if isinstance(index_size, int) else str(index_size)

        st.caption(f"Searched {index_str} records using {strategy_used}")
        st.markdown(f"Found **{len(p2.topk_docs)} matching reports** in {search_ms:.0f}ms")

        # Step 4: Synthesize Answer
        status.update(label="Generating grounded answer...", expanded=True)
        st.write("**Step 4 / 4** — Synthesizing evidence-based answer")
        t4 = time.time()
        answer_text = _synthesize_answer(p2.topk_docs, p1.intent, query)
        t4_ms = (time.time() - t4) * 1000
        st.caption(f"Analyzed top {min(5, len(p2.topk_docs))} reports for patterns")
        st.caption(f"Completed in {t4_ms:.0f}ms")

        total_ms = (time.time() - t0) * 1000
        status.update(label=f"Done — {len(p2.topk_docs)} results in {total_ms:.0f}ms", state="complete", expanded=False)

    # ── Pipeline status line ─────────────────────────────────────────
    st.markdown(f"""
    <div class="pipeline-status">
        Classified as <span class="intent-pill {intent_css}">{intent_label}</span>
        &nbsp;&middot;&nbsp; Searched <strong>{index_str}</strong> records
        &nbsp;&middot;&nbsp; <span class="teal">{len(p2.topk_docs)} matches</span> in {total_ms:.0f}ms
    </div>
    """, unsafe_allow_html=True)

    # ── Answer Card ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-card-header">
            <span>&#128161;</span>
            <span class="answer-card-label">Grounded Answer</span>
        </div>
        <div class="answer-card-body">{answer_text}</div>
        <div class="answer-card-footer">
            All findings are derived from retrieved ASRS incident reports listed below.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Evidence Cards ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="evidence-header">
        <span class="evidence-title">&#128196; Supporting Evidence</span>
        <span class="evidence-count">{len(p2.topk_docs)} incident reports</span>
    </div>
    """, unsafe_allow_html=True)

    for i, doc in enumerate(p2.topk_docs):
        rank = i + 1
        rd_cls = f"rd-{rank}" if rank <= 3 else "rd-n"
        scores = doc.scores
        final = scores.get("final", 0)
        meta = doc.metadata or {}
        event_id = meta.get("event_id", doc.doc_id)

        # Short text (3 lines ~250 chars)
        text_short = doc.chunk_text[:250].strip()
        if len(doc.chunk_text) > 250:
            cut = text_short.rfind(". ")
            if cut > 100:
                text_short = text_short[:cut + 1]
            else:
                text_short += "..."

        # Meta chips
        chips = ""
        for field, label in META_FIELDS:
            val = meta.get(field)
            if val and str(val).strip().lower() not in ("nan", "none", ""):
                chips += f'<span class="ev-chip">{label}: {val}</span>'

        st.markdown(f"""
        <div class="ev-card">
            <div class="ev-top">
                <div class="ev-rank">
                    <span class="rank-dot {rd_cls}">#{rank}</span>
                    <span class="ev-id">Report {event_id}</span>
                </div>
                <span class="ev-match">{final:.0%} relevant</span>
            </div>
            <div class="ev-text">{text_short}</div>
            <div class="ev-meta">{chips}</div>
        </div>
        """, unsafe_allow_html=True)

        # Expandable full text + scores
        with st.expander(f"Show full report & scores — Report {event_id}", expanded=False):
            st.markdown(doc.chunk_text)
            sem = scores.get("semantic", 0)
            bm = scores.get("bm25", 0)
            st.caption(f"Contextual: {sem:.4f} | Keyword: {bm:.4f} | Final: {final:.4f}")
            if meta:
                filtered = {k: v for k, v in meta.items() if k not in ("source", "document_type", "applied_filters", "chunk_index") and str(v).lower() not in ("nan", "none", "")}
                if filtered:
                    st.json(filtered, expanded=False)

    # ── Technical Details ────────────────────────────────────────────
    with st.expander("Technical details (for researchers)"):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(f"**Query (original):** {p1.query_raw}")
            st.markdown(f"**Query (rewritten):** {p1.rewritten_query}")
            st.markdown(f"**Intent source:** `{p1.intent_source}` (confidence: {p1.intent_confidence:.0%})")
            if p1.expanded_queries:
                st.markdown("**Query expansions:**")
                for eq in p1.expanded_queries[:5]:
                    st.caption(f"- {eq}")
        with col_t2:
            st.markdown(f"**Strategy:** `{strategy_used}`")
            st.markdown(f"**Fallback:** `{p1.retrieval_plan.fallback_strategy}`")
            st.markdown(f"**Routing:** {p1.retrieval_plan.routing_reason}")
            st.markdown(f"**Index:** {index_str} vectors")
            st.markdown(f"**Adapter mode:** `{mode}`")
        st.json(diag, expanded=False)

elif search_clicked and not query.strip():
    st.warning("Please enter a search query.")

else:
    # ── Empty State ──────────────────────────────────────────────────
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">&#128752;</div>
        <div class="empty-title">Search aviation incident reports</div>
        <div class="empty-sub">
            Describe a safety concern or incident scenario. The system will find relevant reports
            and provide an evidence-based answer.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="hiw-strip">
        <div class="hiw-step">
            <div class="hiw-num hiw-n1">1</div>
            <div class="hiw-label">Understand your query</div>
            <div class="hiw-desc">Classify intent & optimize search</div>
        </div>
        <div class="hiw-arrow">&#8594;</div>
        <div class="hiw-step">
            <div class="hiw-num hiw-n2">2</div>
            <div class="hiw-label">Find similar reports</div>
            <div class="hiw-desc">Search 111K+ ASRS records</div>
        </div>
        <div class="hiw-arrow">&#8594;</div>
        <div class="hiw-step">
            <div class="hiw-num hiw-n3">3</div>
            <div class="hiw-label">Answer with evidence</div>
            <div class="hiw-desc">Grounded, cited response</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
