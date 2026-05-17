"""
Query execution with error recovery and retry logic.
"""

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from src.database import get_connection
from src.llm_provider import call_llm_json
from src.sql_validator import validate_sql

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
QUERY_TIMEOUT_S = 10


@dataclass
class ExecutionResult:
    success: bool
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    error_message: str = ""
    execution_time_ms: float = 0.0
    row_count: int = 0
    column_count: int = 0
    retries_needed: int = 0
    final_sql: str = ""


_FIX_SYSTEM_PROMPT = """You are an expert SQL debugger for DuckDB.
You are given a SQL query that failed with an error. Fix it.
Return ONLY valid JSON: {"sql": "fixed SQL here", "explanation": "what you fixed"}
"""


def _fix_sql_with_llm(question: str, failed_sql: str, error: str) -> str:
    user_prompt = (
        f"The following DuckDB SQL query failed.\n\n"
        f"ORIGINAL QUESTION: {question}\n\n"
        f"FAILED SQL:\n{failed_sql}\n\n"
        f"ERROR MESSAGE: {error}\n\n"
        "Fix the SQL so it answers the original question correctly. "
        "Use only tables/views that exist: tracks, artists, albums, genres, "
        "key_names, track_artists, track_genres, v_track_details, v_artist_tracks."
    )
    result = call_llm_json(_FIX_SYSTEM_PROMPT, user_prompt)
    return result.get("sql", failed_sql)


def execute_query(question: str, sql: str) -> ExecutionResult:
    """
    Execute validated SQL against DuckDB with automatic retry on failure.
    """
    con = get_connection()
    current_sql = sql
    retries = 0

    for attempt in range(MAX_RETRIES + 1):
        start_ts = time.time()
        try:
            # Set per-query timeout via DuckDB pragma
            con.execute(f"SET threads TO 4")

            df = con.execute(current_sql).df()
            elapsed_ms = (time.time() - start_ts) * 1000

            return ExecutionResult(
                success=True,
                dataframe=df,
                execution_time_ms=round(elapsed_ms, 1),
                row_count=len(df),
                column_count=len(df.columns),
                retries_needed=retries,
                final_sql=current_sql,
            )

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Query attempt {attempt + 1} failed: {error_msg}")

            if attempt >= MAX_RETRIES:
                return ExecutionResult(
                    success=False,
                    error_message=f"Failed after {MAX_RETRIES} retries. Last error: {error_msg}",
                    retries_needed=retries,
                    final_sql=current_sql,
                )

            # Ask LLM to fix the SQL
            logger.info(f"Asking LLM to fix SQL (retry {attempt + 1}/{MAX_RETRIES})...")
            try:
                fixed_sql = _fix_sql_with_llm(question, current_sql, error_msg)
                validation = validate_sql(fixed_sql, con)
                if validation.is_valid:
                    current_sql = validation.sanitized_sql
                    retries += 1
                    logger.info(f"LLM fix validated. Retrying with new SQL.")
                else:
                    logger.warning(
                        f"LLM-fixed SQL still invalid: {validation.errors}. Stopping."
                    )
                    return ExecutionResult(
                        success=False,
                        error_message=(
                            f"Generated SQL could not be fixed. Errors: {'; '.join(validation.errors)}"
                        ),
                        retries_needed=retries,
                        final_sql=current_sql,
                    )
            except Exception as fix_err:
                logger.error(f"LLM fix attempt failed: {fix_err}")
                return ExecutionResult(
                    success=False,
                    error_message=f"Query failed and fix attempt errored: {error_msg}",
                    retries_needed=retries,
                    final_sql=current_sql,
                )
    # Should not reach here
    return ExecutionResult(
        success=False,
        error_message="Unexpected execution flow",
        final_sql=current_sql,
    )
