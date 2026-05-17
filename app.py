"""QueryPilot — AI-Powered Music Analytics"""

import logging
import math
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Persistent history (SQLite) ────────────────────────────────────────────
_HIST_DB = Path(__file__).parent / "data" / "history.db"

def _hist_conn():
    con = sqlite3.connect(_HIST_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            question  TEXT,
            summary   TEXT,
            sql       TEXT,
            narrative TEXT,
            chart_type TEXT,
            confidence REAL,
            ts        TEXT
        )
    """)
    con.commit()
    return con

def _load_hist_db() -> list[dict]:
    try:
        con = _hist_conn()
        rows = con.execute(
            "SELECT question, summary, sql, narrative, chart_type, confidence, ts "
            "FROM history ORDER BY id DESC LIMIT 20"
        ).fetchall()
        con.close()
        return [
            {"question": r[0], "summary": r[1], "sql": r[2],
             "narrative": r[3], "chart_type": r[4], "confidence": r[5],
             "time": r[6], "results": None}
            for r in rows
        ]
    except Exception:
        return []

def _persist_hist(question: str, summary: str, sql: str,
                  narrative: str, chart_type: str, confidence: float):
    try:
        con = _hist_conn()
        con.execute(
            "INSERT INTO history (question, summary, sql, narrative, chart_type, confidence, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (question, summary, sql, narrative, chart_type, confidence,
             datetime.now().strftime("%H:%M:%S")),
        )
        con.commit()
        con.close()
    except Exception:
        pass

def _replay_hist(item: dict) -> dict | None:
    """Re-execute a history item's SQL without calling the LLM."""
    from src.database import get_connection
    from src.visualizer import create_visualization
    try:
        con = get_connection()
        import pandas as pd
        import time as _time
        t0 = _time.perf_counter()
        df = con.execute(item["sql"]).df()
        ms = (_time.perf_counter() - t0) * 1000
        from src.query_executor import ExecutionResult
        ex = ExecutionResult(
            success=True, dataframe=df, final_sql=item["sql"],
            row_count=len(df), column_count=len(df.columns),
            execution_time_ms=ms, retries_needed=0,
        )
        viz = create_visualization(df, item["question"], llm_hint=item.get("chart_type", "table"))
        return {
            "gen": {"sql": item["sql"], "confidence": item.get("confidence", 0),
                    "explanation": "", "chart_recommendation": item.get("chart_type", "table")},
            "validation": None,
            "exec": ex,
            "viz": viz,
            "narrative": item.get("narrative", ""),
        }
    except Exception as e:
        logging.warning(f"History replay failed: {e}")
        return None

st.set_page_config(
    page_title="QueryPilot",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)
logging.basicConfig(level=logging.INFO)

# ── Cached resources ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_db():
    from src.database import DB_PATH, setup_database
    if not DB_PATH.exists():
        setup_database()
    return True

@st.cache_resource(show_spinner=False)
def _schema():
    from src.database import load_schema
    return load_schema()

@st.cache_resource(show_spinner=False)
def _stats():
    from src.database import get_connection
    con = get_connection()
    return (
        con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0],
        con.execute("SELECT COUNT(*) FROM artists").fetchone()[0],
        con.execute("SELECT COUNT(*) FROM genres").fetchone()[0],
    )

# ── SVG waveform ───────────────────────────────────────────────────────────
def _waveform_svg():
    pts = []
    for x in range(0, 1001, 5):
        y = 28 + 13*(0.45*math.sin(x*0.018) + 0.3*math.sin(x*0.047+1.2) + 0.25*math.sin(x*0.093+2.5))
        pts.append(f"{x},{y:.1f}")
    wave = " ".join(pts)
    # Mirrored fill
    mir = " ".join(f"{x},{56-float(pt.split(',')[1]):.1f}" for x, pt in
                   [(p.split(',')[0], p) for p in reversed(pts)])
    return f"""<svg viewBox="0 0 1000 56" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="wg" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0%" stop-color="#1DB954" stop-opacity="0"/>
      <stop offset="30%" stop-color="#1DB954" stop-opacity="0.4"/>
      <stop offset="70%" stop-color="#1DB954" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#1DB954" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="wgf" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="#1DB954" stop-opacity="0.08"/>
      <stop offset="100%" stop-color="#1DB954" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <polygon points="{wave} 1000,56 0,56" fill="url(#wgf)"/>
  <polyline points="{wave}" stroke="url(#wg)" stroke-width="1.8" fill="none"/>
</svg>"""

# ── Session state ──────────────────────────────────────────────────────────
for k, v in [("history", []), ("hist_idx", None), ("last_r", None), ("last_q", ""), ("pending", ""), ("hist_loaded", False), ("auto_submit", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# Load persisted history once per session
if not st.session_state.hist_loaded:
    st.session_state.history = _load_hist_db()
    st.session_state.hist_loaded = True

if st.session_state.pending:
    st.session_state["qbox"] = st.session_state.pending
    st.session_state.pending = ""

# ── Chips config ───────────────────────────────────────────────────────────
# Each question: (label, full_question, complexity_level 1–4)
# 1 = simple aggregation  2 = grouped / filtered
# 3 = multi-join / subquery  4 = window function / CTE
CHIPS = [
    ("GENRE", "#4a4a4a", [
        ("Top 10 genres",        "What are the top 10 genres by number of tracks?", 1),
        ("Most explicit genres", "Which genres have the most explicit tracks?", 1),
        ("Highest energy",       "Which genres have the highest average energy?", 2),
        ("Happiest by valence",  "What are the happiest genres based on average valence score?", 2),
        ("Genre popularity rank","Rank all genres from most to least popular based on average track popularity", 4),
    ]),
    ("ARTISTS", "#4a4a4a", [
        ("Most prolific artists",  "Which artists have the most tracks in the dataset?", 1),
        ("Highest avg popularity", "Which artists have the highest average popularity with at least 5 tracks?", 2),
        ("Multi-genre artists",    "Find artists who appear in more than 3 different genres and list all their genres", 3),
        ("Top track per artist",   "For each artist with at least 10 tracks, show their single most popular track", 4),
    ]),
    ("AUDIO", "#4a4a4a", [
        ("Avg tempo",            "What is the average tempo across all tracks?", 1),
        ("Danceability by genre","What is the average danceability score by genre, ranked highest to lowest?", 2),
        ("Dance vs popularity",  "What is the relationship between danceability and popularity across genres?", 2),
        ("Audio fingerprint",    "Show the full audio fingerprint for the top 5 genres — average energy, valence, acousticness, danceability and tempo", 3),
        ("Popularity quartiles", "Divide all tracks into 4 popularity buckets and show the average danceability and track count per bucket", 4),
    ]),
    ("TRENDS", "#4a4a4a", [
        ("Most popular songs",   "What are the 10 most popular songs in the dataset?", 1),
        ("Longest by genre",     "What is the average song duration in minutes by genre?", 2),
        ("Dance + energy combo", "Which genres have songs that are both highly danceable and highly energetic?", 2),
        ("Artists above average","Which artists have average popularity above the overall dataset average?", 3),
        ("Top 3 per genre",      "Within each genre, what are the top 3 most popular tracks?", 4),
    ]),
]

LEVEL_COLORS = {1: "#1DB954", 2: "#38BDF8", 3: "#F59E0B", 4: "#A855F7"}

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}
#MainMenu, footer, .stDeployButton { visibility: hidden; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1140px !important; }

/* ══════════ HERO ══════════ */
.hero {
    background: linear-gradient(160deg,
        #060d08 0%, #071a0c 25%, #0a2410 50%, #071a0c 75%, #060d08 100%);
    border-radius: 0 0 24px 24px;
    padding: 3.5rem 3.5rem 0;
    position: relative; overflow: hidden;
    margin: 0 -2rem;
    border-bottom: 1px solid #1DB95422;
}
/* Radial glow — top right */
.hero::before {
    content: '';
    position: absolute; top: -120px; right: -100px;
    width: 500px; height: 500px; border-radius: 50%;
    background: radial-gradient(circle, rgba(29,185,84,0.22) 0%, transparent 65%);
    animation: glow 7s ease-in-out infinite;
    pointer-events: none;
}
/* Radial glow — bottom left */
.hero::after {
    content: '';
    position: absolute; bottom: 20px; left: -60px;
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(29,185,84,0.1) 0%, transparent 70%);
    animation: glow 11s ease-in-out infinite reverse;
    pointer-events: none;
}
@keyframes glow {
    0%,100% { transform: scale(1);    opacity: .7; }
    50%      { transform: scale(1.18); opacity: 1;  }
}

.hero-inner { position: relative; z-index: 2; }

.hero-eyebrow {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: #1DB954;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 8px;
}
.hero-eyebrow::before {
    content: '';
    width: 20px; height: 2px;
    background: #1DB954;
    border-radius: 2px;
    display: inline-block;
}

.hero-title {
    font-size: 4rem; font-weight: 900;
    color: #ffffff;
    letter-spacing: -0.045em; line-height: 0.95;
    margin: 0 0 1rem;
}
.hero-title .green {
    color: #1DB954;
    text-shadow: 0 0 40px rgba(29,185,84,0.45), 0 0 80px rgba(29,185,84,0.15);
}

.hero-sub {
    font-size: 1.05rem; color: rgba(255,255,255,0.42);
    margin: 0 0 2rem; line-height: 1.65;
    font-weight: 400; max-width: 480px;
}

.stat-row {
    display: flex; gap: 24px; align-items: center;
    margin-bottom: 2.5rem; flex-wrap: wrap;
}
.stat-item {
    display: flex; flex-direction: column; gap: 2px;
}
.stat-num {
    font-size: 1.5rem; font-weight: 800; color: #fff;
    letter-spacing: -0.03em; line-height: 1;
}
.stat-lbl {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #1DB95480;
}
.stat-sep {
    width: 1px; height: 32px;
    background: linear-gradient(to bottom, transparent, #1DB95430, transparent);
}

.hero-wave {
    margin: 0 -3.5rem;
    display: block;
    position: relative; z-index: 1;
    height: 56px;
}

/* ══════════ SEARCH STRIP ══════════ */
.search-strip {
    background: #0A0A0A;
    padding: 1.6rem 3.5rem 1.2rem;
    margin: 0 -2rem 1.5rem;
    border-bottom: 1px solid #1a1a1a;
}

/* Free every wrapper layer from Streamlit's fixed height */
.stTextArea,
.stTextArea > div,
.stTextArea > div > div {
    height: auto !important;
    min-height: unset !important;
    margin: 0 !important;
    padding: 0 !important;
}
.stTextArea [data-testid="stWidgetLabel"] {
    display: none !important; height: 0 !important; margin: 0 !important;
}
.stTextArea > div > div > textarea {
    background: #111 !important;
    border: 1.5px solid #222 !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 13px 18px !important;
    height: auto !important;
    min-height: 50px !important;
    max-height: 160px !important;
    line-height: 1.5 !important;
    box-sizing: border-box !important;
    resize: none !important;
    overflow-y: hidden !important;
    field-sizing: content !important;
    transition: border-color .2s, box-shadow .2s !important;
    letter-spacing: 0.01em !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: #1DB954 !important;
    box-shadow: 0 0 0 3px rgba(29,185,84,0.12), 0 0 20px rgba(29,185,84,0.06) !important;
    outline: none !important;
    background: #141414 !important;
}
.stTextArea > div > div > textarea::placeholder { color: #383838 !important; }
/* Vertically center the search row so textarea and buttons share the same midline */
[data-testid="stHorizontalBlock"]:has(.stTextArea) {
    align-items: center !important;
}

/* Keep action buttons vertically centered with the textarea */
div[data-testid="column"]:has(button[kind="primary"]),
div[data-testid="column"]:has(button[kind="secondary"]) {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

/* Analyze button */
div[data-testid="column"]:has(button[kind="primary"]) .stButton > button {
    background: #1DB954 !important;
    border: none !important; border-radius: 12px !important;
    color: #000 !important; font-weight: 800 !important;
    font-size: 0.88rem !important; letter-spacing: 0.03em !important;
    height: 50px !important; width: 100% !important;
    box-shadow: 0 0 24px rgba(29,185,84,0.35) !important;
    transition: all .2s !important;
}
div[data-testid="column"]:has(button[kind="primary"]) .stButton > button:hover {
    background: #23e066 !important;
    box-shadow: 0 0 36px rgba(29,185,84,0.55) !important;
    transform: translateY(-1px) !important;
}
/* Clear button */
div[data-testid="column"]:has(button[kind="secondary"]) .stButton > button {
    background: transparent !important; border: 1.5px solid #222 !important;
    border-radius: 12px !important; color: #444 !important;
    height: 50px !important; width: 100% !important;
    font-size: 1.1rem !important; transition: all .18s !important;
}
div[data-testid="column"]:has(button[kind="secondary"]) .stButton > button:hover {
    border-color: #EF4444 !important; color: #EF4444 !important;
    background: rgba(239,68,68,0.06) !important;
}

/* ══════════ CHIPS ══════════ */
.cat-header {
    display: flex; align-items: center; gap: 10px;
    margin: 1.2rem 0 0.6rem;
}
.cat-dot {
    width: 7px; height: 7px; border-radius: 50%;
    flex-shrink: 0;
}
.cat-name {
    font-size: 0.67rem; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase;
}
.cat-line {
    flex: 1; height: 1px;
    background: linear-gradient(to right, currentColor, transparent);
    opacity: 0.15;
}

/* ══════════ CHIPS ══════════ */
/* Marker containers hidden from layout but kept in DOM for adjacent sibling CSS */
.stElementContainer:has([class^="chip-c"]) {
    display: none !important;
}
/* All chip buttons — any button whose column also has a chip-c marker */
.stElementContainer:has([class^="chip-c"]) + .stElementContainer button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid #252525 !important;
    border-radius: 8px !important;
    color: #888 !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    padding: 9px 20px !important;
    line-height: 1.4 !important; white-space: nowrap !important;
    box-shadow: none !important; width: 100% !important;
    text-align: left !important;
    transition: all .15s ease !important;
}
.stElementContainer:has([class^="chip-c"]) + .stElementContainer button:hover {
    background: rgba(29,185,84,0.07) !important;
    border-color: #333 !important;
    color: #e0e0e0 !important;
    transform: translateY(-1px) !important;
}
/* Left border strip per complexity level */
.stElementContainer:has(.chip-c1) + .stElementContainer button {
    border-left: 3px solid #1DB954 !important;
    border-radius: 3px 8px 8px 3px !important;
    padding-left: 12px !important;
}
.stElementContainer:has(.chip-c2) + .stElementContainer button {
    border-left: 3px solid #38BDF8 !important;
    border-radius: 3px 8px 8px 3px !important;
    padding-left: 12px !important;
}
.stElementContainer:has(.chip-c3) + .stElementContainer button {
    border-left: 3px solid #F59E0B !important;
    border-radius: 3px 8px 8px 3px !important;
    padding-left: 12px !important;
}
.stElementContainer:has(.chip-c4) + .stElementContainer button {
    border-left: 3px solid #A855F7 !important;
    border-radius: 3px 8px 8px 3px !important;
    padding-left: 12px !important;
}

.complexity-legend {
    display: flex; gap: 18px; align-items: center;
    margin: 8px 0 0 2px;
}
.cl-item {
    display: flex; align-items: center; gap: 5px;
    font-size: 0.68rem; color: #444; letter-spacing: 0.03em;
    white-space: nowrap;
}
.cl-dot {
    width: 3px; height: 14px; border-radius: 2px; flex-shrink: 0;
}

/* ══════════ PIPELINE ERROR CARDS ══════════ */
.pc-card {
    display: flex; gap: 16px; align-items: flex-start;
    border-radius: 12px; padding: 16px 20px; margin: 1rem 0;
}
.pc-setup {
    background: rgba(251,191,36,0.05);
    border: 1px solid rgba(251,191,36,0.2);
    border-left: 3px solid #FBBF24;
}
.pc-warn {
    background: rgba(251,191,36,0.04);
    border: 1px solid rgba(251,191,36,0.15);
    border-left: 3px solid #F59E0B;
}
.pc-err {
    background: rgba(239,68,68,0.04);
    border: 1px solid rgba(239,68,68,0.15);
    border-left: 3px solid #EF4444;
}
.pc-icon { font-size: 1.3rem; margin-top: 1px; flex-shrink: 0; }
.pc-body { flex: 1; min-width: 0; }
.pc-title {
    font-size: 0.92rem; font-weight: 700; color: #ddd;
    margin-bottom: 6px;
}
.pc-text { font-size: 0.86rem; color: #777; line-height: 1.6; margin: 0 0 8px; }
.pc-step {
    font-size: 0.84rem; color: #666; line-height: 1.6;
    padding: 3px 0;
}
.pc-hint {
    font-size: 0.79rem; color: #444; font-style: italic;
    margin-top: 10px;
}
.pc-card code {
    background: #1c1c1c; padding: 1px 6px; border-radius: 4px;
    font-size: 0.8rem; color: #aaa; font-family: monospace;
}

/* ══════════ CONFIDENCE WARNING ══════════ */
.conf-warning {
    display: flex; align-items: flex-start; gap: 12px;
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.25);
    border-left: 3px solid #FBBF24;
    border-radius: 10px; padding: 12px 16px;
    margin-bottom: 1.2rem;
    font-size: 0.88rem; color: #d4a820;
    line-height: 1.5;
}
.conf-warning-icon { font-size: 1.1rem; margin-top: 1px; flex-shrink: 0; }
.conf-warning-sub { font-size: 0.8rem; color: #8a7020; }

/* ══════════ DIVIDER ══════════ */
.qdiv { border: none; border-top: 1px solid #1a1a1a; margin: 1.8rem 0; }

/* ══════════ RESULTS ══════════ */
.result-label {
    font-size: 0.65rem; font-weight: 800; letter-spacing: 0.16em;
    text-transform: uppercase; color: #1DB954;
    margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;
}
.result-label::before { content:''; width:16px; height:2px; background:#1DB954; border-radius:2px; }
.result-q {
    font-size: 1.6rem; font-weight: 800; color: #fff;
    letter-spacing: -0.03em; line-height: 1.15;
    margin-bottom: 1.4rem;
}

/* Insight card */
.insight-card {
    background: linear-gradient(145deg, #0c1f10 0%, #0d1a0e 100%);
    border: 1px solid #1DB95430;
    border-left: 3px solid #1DB954;
    border-radius: 14px; padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    font-size: 0.95rem; line-height: 1.82; color: #c8d8c8;
    box-shadow: 0 0 40px rgba(29,185,84,0.06), 0 4px 24px rgba(0,0,0,0.3);
    position: relative;
}
.insight-eyebrow {
    font-size: 0.64rem; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: #1DB954;
    margin-bottom: 0.7rem; display: block;
}
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    border-radius: 6px; padding: 2px 9px;
    font-size: 0.7rem; font-weight: 700;
    margin-left: 10px; vertical-align: middle;
}
.badge-high   { background: rgba(29,185,84,0.15); color: #1DB954; border: 1px solid rgba(29,185,84,0.3);}
.badge-medium { background: rgba(251,191,36,0.12); color: #FBBF24; border: 1px solid rgba(251,191,36,0.3);}
.badge-low    { background: rgba(239,68,68,0.1);  color: #F87171; border: 1px solid rgba(239,68,68,0.25);}

/* Cards */
.dark-card {
    background: #111; border: 1px solid #1e1e1e;
    border-radius: 14px; padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-label {
    font-size: 0.63rem; font-weight: 800; letter-spacing: 0.13em;
    text-transform: uppercase; color: #333; margin-bottom: 0.9rem;
}

/* Meta pills */
.meta-row { display:flex; flex-wrap:wrap; gap:6px; padding:1rem 0 0.2rem; border-top:1px solid #1a1a1a; }
.meta-pill {
    background: #111; border: 1px solid #1e1e1e;
    border-radius: 6px; padding: 3px 10px;
    font-size: 0.71rem; font-weight: 500; color: #3a3a3a;
}

/* History */
.hist-item { border-left:2px solid #1e1e1e; padding:6px 0 6px 14px; margin-bottom:12px; transition:.2s; }
.hist-item:hover { border-left-color:#1DB954; }
.hist-q { font-size:.87rem; font-weight:600; color:#ddd; line-height:1.3; }
.hist-s { font-size:.76rem; color:#444; margin-top:3px; line-height:1.4; }
.hist-t { font-size:.67rem; color:#2a2a2a; margin-top:3px; }

/* ══════════ SIDEBAR ══════════ */
[data-testid="stSidebar"] > div:first-child {
    background: #080808 !important;
    border-right: 1px solid #111 !important;
}
[data-testid="stSidebar"] h2 { color: #fff !important; font-size:1rem !important; font-weight:700 !important; }
[data-testid="stSidebar"] .streamlit-expanderHeader {
    font-size: 0.8rem !important; color: #555 !important; font-weight: 500 !important;
    background: #0d0d0d !important; border-radius: 8px !important;
    border: 1px solid #1a1a1a !important;
}
[data-testid="stSidebar"] .streamlit-expanderHeader:hover { color: #1DB954 !important; }
[data-testid="stSidebar"] .streamlit-expanderContent {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}
.prov-card {
    background: #0d0d0d; border: 1px solid #1a1a1a;
    border-radius: 10px; padding: 12px 14px; margin-bottom: 16px;
}
.prov-lbl { font-size:.62rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#2a2a2a; margin-bottom:6px; }
.prov-name { font-size:.86rem; font-weight:600; color:#ccc; }
.prov-calls { font-size:.72rem; color:#333; margin-top:4px; }

/* Status box */
[data-testid="stStatusWidget"] { border-radius: 12px !important; background: #0d0d0d !important; border: 1px solid #1DB95422 !important; }

/* Confidence threshold slider */
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background: #1DB954 !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: #1DB954 !important;
    border-color: #1DB954 !important;
    box-shadow: 0 0 8px rgba(29,185,84,0.4) !important;
}

/* SQL code block — scrollable, max height */
.dark-card [data-testid="stCode"] > div {
    max-height: 280px !important;
    overflow-y: auto !important;
    border-radius: 8px !important;
}
.dark-card [data-testid="stCode"] pre {
    white-space: pre !important;
    word-break: normal !important;
    overflow-x: auto !important;
}

/* Always-visible copy button on code blocks */
[data-testid="stCode"] button {
    opacity: 1 !important;
    background: rgba(29,185,84,0.08) !important;
    border: 1px solid rgba(29,185,84,0.25) !important;
    color: #1DB954 !important;
    border-radius: 6px !important;
    transition: all .15s !important;
}
[data-testid="stCode"] button:hover {
    background: rgba(29,185,84,0.18) !important;
    border-color: rgba(29,185,84,0.5) !important;
}

/* CSV download button in data card */
.dark-card [data-testid="stDownloadButton"] button {
    background: rgba(29,185,84,0.08) !important;
    border: 1px solid rgba(29,185,84,0.25) !important;
    color: #1DB954 !important;
    border-radius: 7px !important;
    font-size: 0.75rem !important; font-weight: 600 !important;
    padding: 4px 10px !important; height: auto !important;
    transition: all .15s !important;
}
.dark-card [data-testid="stDownloadButton"] button:hover {
    background: rgba(29,185,84,0.18) !important;
    border-color: rgba(29,185,84,0.5) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## QueryPilot")
    try:
        from src.llm_provider import get_api_call_count, get_provider_info
        prov = get_provider_info()
        dot = "🟢" if prov["color"] == "green" else "🔴"
        st.markdown(
            f'<div class="prov-card"><div class="prov-lbl">AI Provider</div>'
            f'<div class="prov-name">{dot} {prov["label"]}</div>'
            f'<div class="prov-calls">API calls: {get_api_call_count()}</div></div>',
            unsafe_allow_html=True,
        )
    except EnvironmentError as e:
        st.error(str(e))

    st.markdown("**Schema**")
    try:
        schema = _schema()
        for tname, tdata in schema.get("tables", {}).items():
            with st.expander(tname):
                st.caption(tdata.get("description", ""))
                for col, ci in tdata.get("columns", {}).items():
                    st.markdown(
                        f"<small style='color:#3a3a3a'><b style='color:#555'>{col}</b>"
                        f"<br>&nbsp;&nbsp;{ci.get('description','')}</small><br>",
                        unsafe_allow_html=True,
                    )
        for vname, vdata in schema.get("views", {}).items():
            with st.expander(f"{vname} (view)"):
                st.caption(vdata.get("description", ""))
    except Exception:
        st.caption("Schema loads after DB init.")

    st.markdown("---")
    st.markdown(
        '<div class="prov-lbl" style="margin-bottom:8px">Confidence Threshold</div>',
        unsafe_allow_html=True,
    )
    st.slider(
        "conf_thresh_slider",
        min_value=0.0, max_value=1.0, value=0.70, step=0.05,
        key="conf_threshold",
        label_visibility="collapsed",
        help="Queries below this confidence score will show a warning before you read the results.",
    )
    thresh_pct = int(st.session_state.get("conf_threshold", 0.70) * 100)
    color = "#1DB954" if thresh_pct <= 50 else "#FBBF24" if thresh_pct <= 75 else "#F87171"
    st.markdown(
        f'<div style="font-size:.72rem;color:{color};margin-top:-8px">'
        f'Warn below {thresh_pct}% confidence</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#2a2a2a;line-height:1.6'>QueryPilot converts natural language to SQL, "
        "executes it against 90K Spotify tracks in DuckDB, and returns charts + insights.</small>",
        unsafe_allow_html=True,
    )


# ── DB init ────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading…"):
        _init_db()
    tracks, artists, genres = _stats()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()


# ── Hero ───────────────────────────────────────────────────────────────────
wave_svg = _waveform_svg()
st.markdown(f"""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-eyebrow">AI-Powered Music Analytics</div>
    <div class="hero-title">Query<span class="green">Pilot</span></div>
    <div class="hero-sub">
      Ask anything about your Spotify catalog in plain English.<br>
      Get SQL, interactive charts, and business insights instantly.
    </div>
    <div class="stat-row">
      <div class="stat-item">
        <span class="stat-num">{tracks:,}</span>
        <span class="stat-lbl">Tracks</span>
      </div>
      <div class="stat-sep"></div>
      <div class="stat-item">
        <span class="stat-num">{artists:,}</span>
        <span class="stat-lbl">Artists</span>
      </div>
      <div class="stat-sep"></div>
      <div class="stat-item">
        <span class="stat-num">{genres}</span>
        <span class="stat-lbl">Genres</span>
      </div>
      <div class="stat-sep"></div>
      <div class="stat-item">
        <span class="stat-num" style="font-size:1rem;color:#1DB954">⚡ Multi-LLM</span>
        <span class="stat-lbl">Gemini · GPT-4o · Claude · Llama</span>
      </div>
    </div>
  </div>
  <div class="hero-wave">{wave_svg}</div>
</div>
""", unsafe_allow_html=True)


# ── Search (below hero, dark strip) ───────────────────────────────────────
st.markdown('<div class="search-strip">', unsafe_allow_html=True)
c_in, c_btn, c_clr = st.columns([7.5, 1.5, 0.5])
with c_in:
    question_input = st.text_area(
        "q", placeholder="e.g.  Which artists have the most popular songs?",
        key="qbox", label_visibility="collapsed",
    )
with c_btn:
    submit = st.button("Analyze →", type="primary", use_container_width=True)
with c_clr:
    if st.button("✕", type="secondary", use_container_width=True):
        st.session_state["qbox"] = ""
        st.session_state.hist_idx = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)


# ── Chips ──────────────────────────────────────────────────────────────────
for cat_name, cat_color, questions in CHIPS:
    sorted_qs = sorted(questions, key=lambda x: x[2])
    st.markdown(
        f'<div class="cat-header">'
        f'<span class="cat-dot" style="background:{cat_color}"></span>'
        f'<span class="cat-name" style="color:{cat_color}">{cat_name}</span>'
        f'<span class="cat-line" style="color:{cat_color}"></span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    # Extra trailing column absorbs leftover space so chips don't stretch full width
    ratios = [max(len(lbl) * 5, 40) for lbl, _, _ in sorted_qs] + [20]
    all_cols = st.columns(ratios)
    for col, (label, question, level) in zip(all_cols, sorted_qs):
        with col:
            # Invisible marker div — drives adjacent-sibling CSS for left border color
            st.markdown(f'<div class="chip-c{level}"></div>', unsafe_allow_html=True)
            if st.button(label, key=f"chip_{hash(question)}", use_container_width=True):
                st.session_state["pending"] = question
                st.rerun()

st.markdown("""
<div class="complexity-legend">
  <span class="cl-item"><span class="cl-dot" style="background:#1DB954"></span>Simple</span>
  <span class="cl-item"><span class="cl-dot" style="background:#38BDF8"></span>Grouped</span>
  <span class="cl-item"><span class="cl-dot" style="background:#F59E0B"></span>Multi-join</span>
  <span class="cl-item"><span class="cl-dot" style="background:#A855F7"></span>Window / CTE</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='qdiv'>", unsafe_allow_html=True)


# ── Pipeline ───────────────────────────────────────────────────────────────
import html as _html

def _is_quota_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ["429", "quota", "resource_exhausted", "too many requests", "rate limit"])

def _is_no_key_error(e: Exception) -> bool:
    s = str(e).lower()
    return "no llm api key" in s or "no api key" in s

def _is_all_exhausted(e: Exception) -> bool:
    return _is_no_key_error(e) and "all providers exhausted" in str(e).lower()

def _error_card(e: Exception) -> str:
    """Return a styled HTML card for a pipeline exception."""
    if _is_all_exhausted(e):
        return (
            '<div class="pc-card pc-warn">'
            '<div class="pc-icon">⏳</div>'
            '<div class="pc-body">'
            '<div class="pc-title">Daily Limits Reached</div>'
            '<p class="pc-text">All configured free-tier providers have hit their daily token limits '
            '(Google: 20 req/day · Groq: 100K tokens/day). Limits reset at midnight UTC.</p>'
            '<div class="pc-step">→ <strong>Wait until tomorrow</strong> — limits auto-reset overnight</div>'
            '<div class="pc-step">→ Add <code>ANTHROPIC_API_KEY</code> or <code>OPENAI_API_KEY</code> '
            'to <code>.env</code> for unlimited usage today</div>'
            '<div class="pc-hint">Restart after adding a new key: <code>streamlit run app.py</code></div>'
            '</div></div>'
        )
    if _is_no_key_error(e):
        return (
            '<div class="pc-card pc-setup">'
            '<div class="pc-icon">🔑</div>'
            '<div class="pc-body">'
            '<div class="pc-title">API Key Required</div>'
            '<p class="pc-text">QueryPilot needs an LLM provider key to generate SQL. '
            'Both options below are <strong>free</strong>:</p>'
            '<div class="pc-step">① <strong>Gemini</strong> — get a key at '
            '<code>aistudio.google.com</code>, then add '
            '<code>GOOGLE_API_KEY=…</code> to your <code>.env</code></div>'
            '<div class="pc-step">② <strong>Groq</strong> (no daily cap) — '
            '<code>console.groq.com</code> → add <code>GROQ_API_KEY=…</code> to <code>.env</code></div>'
            '<div class="pc-hint">Restart after adding the key: <code>streamlit run app.py</code></div>'
            '</div></div>'
        )
    if _is_quota_error(e):
        return (
            '<div class="pc-card pc-warn">'
            '<div class="pc-icon">⏱</div>'
            '<div class="pc-body">'
            '<div class="pc-title">Rate Limit Reached</div>'
            '<p class="pc-text">The free-tier limit was hit. '
            'The app automatically falls back to other providers when one is exhausted.</p>'
            '<div class="pc-step">→ Wait ~1 min and retry — per-minute limits reset quickly</div>'
            '<div class="pc-step">→ Add <code>GROQ_API_KEY</code> to <code>.env</code> '
            'for unlimited free usage with no daily cap</div>'
            '</div></div>'
        )
    return (
        '<div class="pc-card pc-err">'
        '<div class="pc-icon">⚠</div>'
        '<div class="pc-body">'
        '<div class="pc-title">Something went wrong</div>'
        f'<p class="pc-text">{_html.escape(str(e))}</p>'
        '</div></div>'
    )

def _explain_card(explanation: str) -> str:
    """Wrap an LLM plain-English explanation in a neutral error card."""
    return (
        '<div class="pc-card pc-err">'
        '<div class="pc-icon">💬</div>'
        '<div class="pc-body">'
        '<div class="pc-title">Query could not be processed</div>'
        f'<p class="pc-text">{_html.escape(explanation)}</p>'
        '</div></div>'
    )

def _explain_error(question: str, sql: str, raw_error: str) -> str:
    """Ask the LLM to explain a SQL/validation error in plain English."""
    from src.llm_provider import call_llm
    system = (
        "You are a concise SQL helper for a Spotify analytics app. "
        "A user asked a question in natural language, SQL was generated, but it failed. "
        "In 2 sentences max: (1) explain what went wrong in plain English, no jargon. "
        "(2) suggest a specific rephrasing the user can try. "
        "The dataset has these key columns: genre_name, artist_name, track_name, album_name, "
        "popularity (0-100), danceability, energy, valence, tempo, acousticness, "
        "duration_ms, explicit. Use these names in your suggestion."
    )
    user = f"Question: {question}\nSQL attempted: {sql}\nError: {raw_error}"
    try:
        return call_llm(system, user, temperature=0.2)
    except Exception:
        return raw_error

def run_pipeline(question: str) -> dict | None:
    from src.database import get_connection
    from src.narrator import generate_narrative
    from src.query_executor import execute_query
    from src.sql_generator import generate_sql
    from src.sql_validator import validate_sql
    from src.visualizer import create_visualization

    R = {}
    pipeline_error: str | None = None

    with st.status("Analyzing…", expanded=True) as status:
        st.write("🧠 Generating SQL…")
        try:
            gen = generate_sql(question); R["gen"] = gen
            st.write(f"✅ SQL ready — **{gen['confidence']:.0%}** confidence")
        except Exception as e:
            label = "⚠️ Quota exceeded" if _is_quota_error(e) else "❌ Failed"
            status.update(label=label, state="error")
            st.session_state["_pipeline_err"] = _error_card(e)
            return None

        st.write("🛡️ Validating query…")
        try:
            con = get_connection(); val = validate_sql(gen["sql"], con); R["validation"] = val
            if not val.is_valid:
                raw = "; ".join(val.errors)
                status.update(label="❌ Blocked — unsafe query", state="error")
                st.write("💬 Explaining…")
                st.session_state["_pipeline_err"] = _explain_card(_explain_error(question, gen["sql"], raw))
                return None
            st.write(f"✅ Safe — complexity {val.complexity_score}/5")
        except Exception as e:
            status.update(label="❌ Validation error", state="error")
            st.session_state["_pipeline_err"] = _error_card(e)
            return None

        st.write("⚡ Executing against DuckDB…")
        try:
            ex = execute_query(question, val.sanitized_sql); R["exec"] = ex
            if not ex.success:
                status.update(label="❌ Query failed", state="error")
                st.write("💬 Explaining…")
                st.session_state["_pipeline_err"] = _explain_card(_explain_error(question, val.sanitized_sql, ex.error_message))
                return None
            rt = f", {ex.retries_needed} retries" if ex.retries_needed else ""
            st.write(f"✅ **{ex.row_count:,} rows** in {ex.execution_time_ms:.0f}ms{rt}")
        except Exception as e:
            status.update(label="❌ Execution error", state="error")
            st.session_state["_pipeline_err"] = _error_card(e)
            return None

        st.write("📊 Selecting chart…")
        try:
            viz = create_visualization(ex.dataframe, question, llm_hint=gen["chart_recommendation"])
            R["viz"] = viz
            st.write(f"✅ {viz.chart_type.replace('_',' ').title()}")
        except Exception as e:
            R["viz"] = None; st.write(f"⚠️ {e}")

        st.write("💡 Writing insight…")
        try:
            ct = R.get("viz").chart_type if R.get("viz") else "table"
            R["narrative"] = generate_narrative(question, val.sanitized_sql, ex.dataframe, ct)
            st.write("✅ Done")
        except Exception as e:
            R["narrative"] = "Insight unavailable."

        status.update(label="✨ Complete!", state="complete")
    return R


def display_results(R: dict, question: str) -> None:
    if not R:
        return
    gen = R.get("gen", {}); val = R.get("validation"); ex = R.get("exec")
    viz = R.get("viz"); narrative = R.get("narrative", ""); conf = gen.get("confidence", 0)

    st.markdown(
        f'<div class="result-label">Results</div>'
        f'<div class="result-q">{question}</div>',
        unsafe_allow_html=True,
    )

    bc = "badge-high" if conf > 0.8 else ("badge-medium" if conf > 0.5 else "badge-low")
    bt = ("● High" if conf > 0.8 else "● Medium" if conf > 0.5 else "● Low") + " Confidence"
    st.markdown(
        f'<div class="insight-card">'
        f'<span class="insight-eyebrow">Key Insight</span>'
        f'{narrative}'
        f'<span class="badge {bc}">{bt}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Confidence threshold warning
    threshold = st.session_state.get("conf_threshold", 0.70)
    if conf < threshold:
        st.markdown(
            f'<div class="conf-warning">'
            f'<span class="conf-warning-icon">⚠</span>'
            f'<div><strong>Low confidence result</strong> ({conf:.0%} — your threshold is {threshold:.0%})<br>'
            f'<span class="conf-warning-sub">The SQL may not fully capture your intent. '
            f'Try rephrasing or check the generated SQL below before acting on this.</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if viz and viz.figure is not None:
        viz.figure.update_layout(title_font_color="#FFFFFF", title_font_size=14)
        st.plotly_chart(viz.figure, use_container_width=True, config={"displayModeBar": False})

    cl, cr = st.columns(2, gap="medium")
    with cl:
        st.markdown('<div class="dark-card"><div class="card-label">Data</div>', unsafe_allow_html=True)
        if ex and not ex.dataframe.empty:
            st.dataframe(ex.dataframe, use_container_width=True,
                         height=min(320, (len(ex.dataframe)+1)*35+40))
            dl_col, cap_col = st.columns([1, 3])
            with dl_col:
                st.download_button(
                    "⬇ CSV",
                    ex.dataframe.to_csv(index=False),
                    file_name=f"querypilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with cap_col:
                st.caption(f"{ex.row_count:,} rows × {ex.column_count} columns")
        st.markdown('</div>', unsafe_allow_html=True)

    with cr:
        st.markdown('<div class="dark-card"><div class="card-label">Generated SQL</div>', unsafe_allow_html=True)
        raw_sql = ex.final_sql if ex else gen.get("sql", "")
        try:
            import sqlglot
            formatted_sql = sqlglot.transpile(raw_sql, read="duckdb", pretty=True)[0]
        except Exception:
            formatted_sql = raw_sql
        st.code(formatted_sql, language="sql")
        if gen.get("explanation"):
            st.caption(gen["explanation"])
        st.markdown('</div>', unsafe_allow_html=True)

    if ex and val:
        from src.llm_provider import get_provider_info
        pn = get_provider_info().get("label", "")
        cl2 = viz.chart_type.replace("_", " ").title() if viz else "Table"
        st.markdown(
            f'<div class="meta-row">'
            f'<span class="meta-pill">⏱ {ex.execution_time_ms:.0f}ms</span>'
            f'<span class="meta-pill">📦 {ex.row_count:,} rows</span>'
            f'<span class="meta-pill">🔄 {ex.retries_needed} retries</span>'
            f'<span class="meta-pill">🎯 {conf:.0%} confidence</span>'
            f'<span class="meta-pill">📊 {cl2}</span>'
            f'<span class="meta-pill">⚙️ Complexity {val.complexity_score}/5</span>'
            f'<span class="meta-pill">🤖 {pn}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def save_hist(question: str, R: dict):
    n = R.get("narrative", "")
    summary = (n[:140]+"…") if len(n) > 140 else n
    sql = R.get("exec").final_sql if R.get("exec") else R.get("gen", {}).get("sql", "")
    chart_type = R.get("viz").chart_type if R.get("viz") else "table"
    confidence = R.get("gen", {}).get("confidence", 0)
    item = {
        "question":   question,
        "summary":    summary,
        "sql":        sql,
        "narrative":  n,
        "chart_type": chart_type,
        "confidence": confidence,
        "time":       datetime.now().strftime("%H:%M:%S"),
        "results":    R,
    }
    st.session_state.history.insert(0, item)
    st.session_state.history = st.session_state.history[:20]
    _persist_hist(question, summary, sql, n, chart_type, confidence)


# ── Run ────────────────────────────────────────────────────────────────────
active_q = (st.session_state.get("qbox") or "").strip()
auto_submit = st.session_state.pop("auto_submit", False)

if (submit or auto_submit) and active_q:
    st.session_state.hist_idx = None
    st.session_state.pop("_pipeline_err", None)
    try:
        R = run_pipeline(active_q)
        if R:
            save_hist(active_q, R)
            st.session_state.last_r = R
            st.session_state.last_q = active_q
            display_results(R, active_q)
        elif st.session_state.get("_pipeline_err"):
            st.markdown(st.session_state.pop("_pipeline_err"), unsafe_allow_html=True)
    except Exception:
        logging.error(traceback.format_exc())
        st.markdown(_error_card(Exception("Unexpected error — try rephrasing your question.")), unsafe_allow_html=True)
elif submit:
    st.warning("Type a question or click a chip above.")
else:
    sr, sq = None, ""
    if st.session_state.hist_idx is not None and st.session_state.history:
        idx = st.session_state.hist_idx
        if 0 <= idx < len(st.session_state.history):
            e = st.session_state.history[idx]
            if e.get("results"):
                sr, sq = e["results"], e["question"]
            else:
                # Persisted item — replay SQL without LLM
                sr = _replay_hist(e)
                sq = e["question"]
                if sr:
                    st.session_state.history[idx]["results"] = sr
    elif st.session_state.last_r:
        sr, sq = st.session_state.last_r, st.session_state.last_q
    if sr:
        display_results(sr, sq)


# ── History ────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<hr class='qdiv'>", unsafe_allow_html=True)
    st.markdown("**Recent Queries**")
    for i, e in enumerate(st.session_state.history):
        c1, c2 = st.columns([1, 11])
        with c1:
            if st.button("↩", key=f"h{i}", help="Reload"):
                st.session_state.hist_idx = i
                st.rerun()
        with c2:
            st.markdown(
                f'<div class="hist-item"><div class="hist-q">{e["question"]}</div>'
                f'<div class="hist-s">{e["summary"]}</div>'
                f'<div class="hist-t">{e["time"]}</div></div>',
                unsafe_allow_html=True,
            )
