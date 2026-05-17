"""
Business narrative generator: turns query results into plain-English insights.
"""

import logging

import pandas as pd

from src.llm_provider import call_llm

logger = logging.getLogger(__name__)

_NARRATOR_SYSTEM_PROMPT = """You are a sharp music industry business analyst.
Your job is to turn data query results into crisp, executive-ready insights.

WRITING RULES:
1. Lead with the single most important finding — don't bury it.
2. Include the top 1-2 specific numbers (e.g., "Pop has 8,212 tracks, 23% of the catalog").
3. Note any surprising trend, outlier, or pattern worth calling out.
4. End with a "so what" — one actionable implication for a label, platform, or artist.
5. Tone: confident, direct, VP-level. No jargon. No hedging. No "it appears that."
6. Length: 3-5 sentences max. Quality over quantity.
7. Do NOT repeat the question back. Start immediately with the insight.
"""


def generate_narrative(
    question: str,
    sql: str,
    df: pd.DataFrame,
    chart_type: str,
) -> str:
    """
    Generate a business narrative for the query results.
    Returns a plain-English insight string.
    """
    if df.empty:
        return "The query returned no results. Try broadening your search criteria."

    # Prepare a compact data summary for the LLM
    n_rows = len(df)
    preview_rows = min(20, n_rows)
    data_md = df.head(preview_rows).to_markdown(index=False)

    user_prompt = f"""QUESTION: {question}

DATA ({n_rows} rows total, showing first {preview_rows}):
{data_md}

VISUALIZATION TYPE: {chart_type}

Write a 3-5 sentence business insight about what this data reveals. Lead with the key finding."""

    import time
    for attempt in range(3):
        try:
            narrative = call_llm(_NARRATOR_SYSTEM_PROMPT, user_prompt, temperature=0.3)
            return narrative.strip()
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(k in err for k in ["rate", "quota", "429", "exhausted"])
            if is_rate_limit and attempt < 2:
                wait = (attempt + 1) * 5
                logger.warning(f"Rate limited, retrying narrator in {wait}s…")
                time.sleep(wait)
            else:
                logger.error(f"Narrative generation failed: {e}")
                return _fallback_narrative(question, df)
    return _fallback_narrative(question, df)


def _fallback_narrative(question: str, df: pd.DataFrame) -> str:
    """Simple rule-based fallback when LLM is unavailable."""
    n_rows = len(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    parts = [f"The query returned {n_rows:,} result{'s' if n_rows != 1 else ''}."]

    if numeric_cols:
        col = numeric_cols[0]
        parts.append(
            f"The '{col}' column ranges from {df[col].min():,.2f} to {df[col].max():,.2f} "
            f"with a mean of {df[col].mean():,.2f}."
        )

    if n_rows > 0:
        top_row = df.iloc[0]
        parts.append(f"The top result: {', '.join(str(v) for v in top_row.values[:3])}.")

    return " ".join(parts)
