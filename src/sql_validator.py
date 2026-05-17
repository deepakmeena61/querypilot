"""
Multi-layer SQL validation pipeline.
Layer 1: Blocklist keyword check
Layer 2: SQLGlot parsing + table/column verification
Layer 3: Safety checks (SELECT only, reasonable LIMIT, no cartesian joins)
Layer 4: DuckDB EXPLAIN dry run
"""

import logging
import re
from typing import TYPE_CHECKING

import sqlglot
import sqlglot.expressions as exp
from pydantic import BaseModel

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

# Known tables and their columns
KNOWN_TABLES: dict[str, set[str]] = {
    "tracks": {
        "track_id", "track_name", "album_id", "popularity", "duration_ms", "explicit",
        "danceability", "energy", "key", "loudness", "mode", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo",
        "time_signature", "duration_min",
    },
    "artists": {"artist_id", "artist_name"},
    "albums": {"album_id", "album_name"},
    "genres": {"genre_id", "genre_name"},
    "key_names": {"key", "key_name"},
    "track_artists": {"track_id", "artist_id"},
    "track_genres": {"track_id", "genre_id"},
    "v_track_details": {
        "track_id", "track_name", "album_name", "popularity", "duration_ms", "duration_min",
        "explicit", "danceability", "energy", "key", "key_name", "loudness", "mode",
        "mode_name", "speechiness", "acousticness", "instrumentalness", "liveness",
        "valence", "tempo", "time_signature", "genre_name",
    },
    "v_artist_tracks": {
        "artist_name", "track_id", "track_name", "album_name", "popularity",
        "genre_name", "danceability", "energy", "valence", "tempo", "duration_min",
    },
}

BLOCKED_KEYWORDS = [
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bDROP\b",
    r"\bALTER\b", r"\bCREATE\b", r"\bTRUNCATE\b", r"\bGRANT\b",
    r"\bREVOKE\b", r"\bEXEC\b", r"\bEXECUTE\b",
    r"xp_", r"sp_",
    r"/\*",        # block comment start
    r"--",         # line comment (potential injection)
]


class ValidationResult(BaseModel):
    is_valid: bool
    errors: list[str] = []
    warnings: list[str] = []
    sanitized_sql: str = ""
    tables_used: list[str] = []
    complexity_score: int = 1


def _layer1_blocklist(sql: str) -> list[str]:
    """Reject dangerous SQL keywords."""
    errors = []
    upper = sql.upper()
    for pattern in BLOCKED_KEYWORDS:
        if re.search(pattern, upper, re.IGNORECASE):
            errors.append(f"Blocked keyword detected: {pattern.strip(r'\\b')}")
    # Prevent SQL chaining via semicolons
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        errors.append("Multiple statements detected (semicolon injection attempt)")
    return errors


def _layer2_sqlglot(sql: str) -> tuple[list[str], list[str], list[str]]:
    """Parse with SQLGlot and validate table/column references."""
    errors = []
    warnings = []
    tables_used = []

    try:
        parsed = sqlglot.parse_one(sql, dialect="duckdb")
    except sqlglot.errors.ParseError as e:
        return [f"SQL parse error: {e}"], [], []

    # Extract table references
    for table in parsed.find_all(exp.Table):
        table_name = table.name.lower()
        if table_name:
            tables_used.append(table_name)

    # Validate referenced tables exist
    for tbl in tables_used:
        if tbl not in KNOWN_TABLES and not tbl.startswith("_"):
            warnings.append(f"Unknown table referenced: '{tbl}' — query may fail")

    # Check for obvious invalid column references (only if we can determine the table)
    # We do a lighter check here — full column validation happens in layer 4 (EXPLAIN)
    for col in parsed.find_all(exp.Column):
        col_name = col.name.lower() if col.name else None
        table_ref = col.table.lower() if col.table else None
        if col_name and table_ref and table_ref in KNOWN_TABLES:
            if col_name not in KNOWN_TABLES[table_ref]:
                # Check if it's used as an alias — hard to detect statically, so warn
                warnings.append(
                    f"Column '{col_name}' not found in table '{table_ref}' — verify alias usage"
                )

    return errors, warnings, list(set(tables_used))


def _layer3_safety(sql: str, tables_used: list[str]) -> tuple[list[str], list[str], str, int]:
    """Check query type, limits, and complexity."""
    errors = []
    warnings = []
    sanitized = sql.strip().rstrip(";")

    try:
        parsed = sqlglot.parse_one(sanitized, dialect="duckdb")
    except Exception:
        return errors, warnings, sanitized, 1

    # Must be SELECT or WITH (CTE)
    if not isinstance(parsed, (exp.Select, exp.With)):
        errors.append("Only SELECT queries are allowed")
        return errors, warnings, sanitized, 1

    # Complexity scoring
    join_count = len(list(parsed.find_all(exp.Join)))
    subquery_count = len(list(parsed.find_all(exp.Subquery)))
    complexity = min(5, 1 + join_count + subquery_count * 2)

    if join_count > 5:
        warnings.append(f"High join count ({join_count}) — query may be slow")

    # Check for cartesian joins (join without ON clause)
    for join in parsed.find_all(exp.Join):
        if join.args.get("on") is None and join.args.get("using") is None:
            kind = join.args.get("kind", "")
            if str(kind).upper() not in ("CROSS", ""):
                warnings.append("JOIN without ON condition detected (possible cartesian join)")

    # Check/add LIMIT for non-aggregated queries
    has_limit = parsed.find(exp.Limit) is not None
    has_group = parsed.find(exp.Group) is not None
    has_agg = any(
        isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min))
        for node in parsed.walk()
    )

    if not has_limit and not has_group and not has_agg:
        sanitized = sanitized + " LIMIT 500"
        warnings.append("Added LIMIT 500 to prevent large result sets")

    if complexity >= 4:
        warnings.append(f"Complex query (score {complexity}/5) — may take longer to execute")

    return errors, warnings, sanitized, complexity


def _layer4_explain(sql: str, con: "duckdb.DuckDBPyConnection") -> list[str]:
    """Dry-run the query using DuckDB EXPLAIN."""
    errors = []
    try:
        con.execute(f"EXPLAIN {sql}")
    except Exception as e:
        errors.append(f"Query validation failed: {e}")
    return errors


def validate_sql(
    sql: str, con: "duckdb.DuckDBPyConnection"
) -> ValidationResult:
    """Run the full 4-layer validation pipeline."""
    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Layer 1
    errors1 = _layer1_blocklist(sql)
    all_errors.extend(errors1)
    if errors1:
        return ValidationResult(
            is_valid=False,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_sql=sql,
            tables_used=[],
            complexity_score=1,
        )

    # Layer 2
    errors2, warnings2, tables_used = _layer2_sqlglot(sql)
    all_errors.extend(errors2)
    all_warnings.extend(warnings2)
    if errors2:
        return ValidationResult(
            is_valid=False,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_sql=sql,
            tables_used=tables_used,
            complexity_score=1,
        )

    # Layer 3
    errors3, warnings3, sanitized, complexity = _layer3_safety(sql, tables_used)
    all_errors.extend(errors3)
    all_warnings.extend(warnings3)
    if errors3:
        return ValidationResult(
            is_valid=False,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_sql=sanitized,
            tables_used=tables_used,
            complexity_score=complexity,
        )

    # Layer 4 — only if no hard errors so far
    errors4 = _layer4_explain(sanitized, con)
    all_errors.extend(errors4)

    return ValidationResult(
        is_valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
        sanitized_sql=sanitized,
        tables_used=tables_used,
        complexity_score=complexity,
    )
