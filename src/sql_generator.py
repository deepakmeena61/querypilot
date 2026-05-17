"""
Schema-aware SQL generation from natural language questions.
"""

import json
import logging

from src.database import load_schema
from src.llm_provider import call_llm_json

logger = logging.getLogger(__name__)

_schema_cache: dict | None = None
_system_prompt_cache: str | None = None

FEW_SHOT_EXAMPLES = """
EXAMPLES (natural language → SQL):

Q: How many tracks are in the dataset?
A: {"sql": "SELECT COUNT(*) AS total_tracks FROM tracks", "explanation": "Counts all rows in the tracks table", "chart_recommendation": "kpi", "confidence": 1.0}

Q: What are the top 10 genres by number of tracks?
A: {"sql": "SELECT genre_name, COUNT(*) AS track_count FROM v_track_details GROUP BY genre_name ORDER BY track_count DESC LIMIT 10", "explanation": "Groups tracks by genre and counts them, returning the top 10", "chart_recommendation": "bar", "confidence": 0.98}

Q: What is the average danceability by genre?
A: {"sql": "SELECT genre_name, ROUND(AVG(danceability), 3) AS avg_danceability FROM v_track_details GROUP BY genre_name ORDER BY avg_danceability DESC", "explanation": "Computes average danceability score per genre", "chart_recommendation": "horizontal_bar", "confidence": 0.97}

Q: What are the most popular rock songs?
A: {"sql": "SELECT track_name, album_name, popularity FROM v_track_details WHERE genre_name = 'rock' ORDER BY popularity DESC LIMIT 20", "explanation": "Filters to rock genre and sorts by popularity descending", "chart_recommendation": "table", "confidence": 0.95}

Q: Which artists have the most tracks in the dataset?
A: {"sql": "SELECT artist_name, COUNT(*) AS track_count FROM v_artist_tracks GROUP BY artist_name ORDER BY track_count DESC LIMIT 10", "explanation": "Counts tracks per artist using the artist_tracks view", "chart_recommendation": "horizontal_bar", "confidence": 0.97}

Q: What is the average song duration in minutes by genre?
A: {"sql": "SELECT genre_name, ROUND(AVG(duration_min), 2) AS avg_duration_min FROM v_track_details GROUP BY genre_name ORDER BY avg_duration_min DESC", "explanation": "Averages the pre-computed duration_min column grouped by genre", "chart_recommendation": "horizontal_bar", "confidence": 0.96}

Q: Do louder songs tend to be more energetic?
A: {"sql": "SELECT ROUND(loudness, 0) AS loudness_bucket, ROUND(AVG(energy), 3) AS avg_energy, COUNT(*) AS track_count FROM v_track_details GROUP BY loudness_bucket ORDER BY loudness_bucket", "explanation": "Buckets songs by loudness level and shows average energy per bucket", "chart_recommendation": "scatter", "confidence": 0.85}

Q: Which genres have songs that are both highly danceable and highly energetic?
A: {"sql": "SELECT genre_name, COUNT(*) AS track_count, ROUND(AVG(popularity), 1) AS avg_popularity FROM v_track_details WHERE danceability > 0.7 AND energy > 0.7 GROUP BY genre_name ORDER BY track_count DESC LIMIT 10", "explanation": "Filters to high danceability AND energy tracks, then groups by genre", "chart_recommendation": "bar", "confidence": 0.93}

Q: What is the distribution of song tempos?
A: {"sql": "SELECT CASE WHEN tempo < 80 THEN 'Slow (<80 BPM)' WHEN tempo < 100 THEN 'Moderate (80-100)' WHEN tempo < 120 THEN 'Upbeat (100-120)' WHEN tempo < 140 THEN 'Fast (120-140)' ELSE 'Very Fast (140+)' END AS tempo_range, COUNT(*) AS track_count FROM v_track_details GROUP BY tempo_range ORDER BY MIN(tempo)", "explanation": "Classifies tracks into BPM buckets to show the tempo distribution", "chart_recommendation": "bar", "confidence": 0.9}

Q: Which artists have the highest average popularity?
A: {"sql": "SELECT artist_name, ROUND(AVG(popularity), 1) AS avg_popularity, COUNT(*) AS track_count FROM v_artist_tracks GROUP BY artist_name HAVING COUNT(*) >= 5 ORDER BY avg_popularity DESC LIMIT 15", "explanation": "Calculates average popularity per artist, requiring at least 5 tracks for statistical reliability", "chart_recommendation": "horizontal_bar", "confidence": 0.92}
"""


def _build_system_prompt(schema: dict) -> str:
    tables_info = []
    for table_name, table_data in schema.get("tables", {}).items():
        col_lines = []
        for col_name, col_info in table_data.get("columns", {}).items():
            sample = col_info.get("sample_values", [])[:2]
            desc = col_info.get("description", "")
            col_lines.append(f"    - {col_name} ({col_info['type']}): {desc}. Samples: {sample}")
        tables_info.append(
            f"TABLE: {table_name}\n  Description: {table_data.get('description', '')}\n"
            + "\n".join(col_lines)
        )

    views_info = []
    for view_name, view_data in schema.get("views", {}).items():
        col_names = list(view_data.get("columns", {}).keys())
        views_info.append(
            f"VIEW: {view_name}\n  Description: {view_data.get('description', '')}\n"
            f"  Columns: {', '.join(col_names)}"
        )

    rels = schema.get("relationships", [])
    rel_lines = [f"  {r['from']} → {r['to']}" for r in rels]

    schema_text = (
        "DATABASE SCHEMA:\n\n"
        + "\n\n".join(tables_info)
        + "\n\nVIEWS (use these for most queries):\n\n"
        + "\n\n".join(views_info)
        + "\n\nFOREIGN KEY RELATIONSHIPS:\n"
        + "\n".join(rel_lines)
    )

    return f"""You are an expert SQL analyst for a Spotify music analytics application.
Your job is to convert natural language questions into precise DuckDB SQL queries.

{schema_text}

AUDIO FEATURE REFERENCE:
- danceability (0-1): How suitable a track is for dancing. 0=least danceable, 1=most danceable.
- energy (0-1): Perceptual intensity and activity. 0=calm/slow, 1=fast/loud/noisy.
- loudness (dB): Typical range -60 to 0. Higher (closer to 0) means louder.
- valence (0-1): Musical positiveness. 0=sad/angry/tense, 1=happy/cheerful/euphoric.
- tempo (BPM): Speed of the track in beats per minute. Typical songs 60-200 BPM.
- acousticness (0-1): Confidence the track is acoustic. 1=definitely acoustic.
- instrumentalness (0-1): Probability track has no vocals. >0.5=likely instrumental.
- speechiness (0-1): Presence of spoken words. >0.66=mostly spoken word.
- liveness (0-1): Probability of live recording. >0.8=likely live.

RULES:
1. Only generate SELECT statements. NEVER use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE.
2. Use v_track_details for most queries (track attributes, audio features, genres).
3. Use v_artist_tracks when the query involves artist names or artist statistics.
4. Always use descriptive column aliases (e.g., AS track_count, not just COUNT(*)).
5. Limit results: add LIMIT 20 for ranked lists unless user specifies. Max 500 for raw data.
6. For aggregation queries, no LIMIT needed unless ranking.
7. Use ROUND() for float averages (2-3 decimal places).
8. Genre values use lowercase with hyphens (e.g., 'hip-hop', 'r-n-b', 'drum-and-bass').
9. Respond with JSON only matching the schema: {{sql, explanation, chart_recommendation, confidence}}.

CHART RECOMMENDATIONS:
- "bar": categorical + numeric, <20 categories, short labels
- "horizontal_bar": categorical + numeric, many categories or long labels
- "line": time series or ordered sequence
- "pie": proportions/composition, <8 categories
- "scatter": two numeric columns, correlation analysis
- "heatmap": two categoricals + one numeric
- "kpi": single number result
- "table": complex multi-column results, no clear chart fits

{FEW_SHOT_EXAMPLES}

Respond ONLY with valid JSON in this exact format:
{{
  "sql": "SELECT ...",
  "explanation": "Brief explanation of what the query does",
  "chart_recommendation": "bar|horizontal_bar|line|pie|scatter|heatmap|kpi|table",
  "confidence": 0.0-1.0
}}"""


def get_system_prompt() -> str:
    global _schema_cache, _system_prompt_cache
    if _system_prompt_cache is None:
        _schema_cache = load_schema()
        _system_prompt_cache = _build_system_prompt(_schema_cache)
    return _system_prompt_cache


def generate_sql(question: str) -> dict:
    """
    Convert a natural language question into SQL.

    Returns:
        dict with keys: sql, explanation, chart_recommendation, confidence
    """
    system_prompt = get_system_prompt()
    user_prompt = f"Convert this question to SQL: {question}"

    response_schema = {
        "type": "object",
        "properties": {
            "sql": {"type": "string"},
            "explanation": {"type": "string"},
            "chart_recommendation": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["sql", "explanation", "chart_recommendation", "confidence"],
    }

    result = call_llm_json(system_prompt, user_prompt, response_schema=response_schema)

    # Ensure all expected keys are present
    return {
        "sql": result.get("sql", ""),
        "explanation": result.get("explanation", ""),
        "chart_recommendation": result.get("chart_recommendation", "table"),
        "confidence": float(result.get("confidence", 0.5)),
    }
