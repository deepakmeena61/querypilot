"""
Evaluation engine: runs benchmark questions through the pipeline and scores results.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
TEST_QUESTIONS_PATH = BENCHMARKS_DIR / "test_questions.json"
REPORT_PATH = BENCHMARKS_DIR / "evaluation_report.md"


@dataclass
class QuestionResult:
    id: int
    question: str
    difficulty: str
    category: str
    expected_sql: str
    generated_sql: str = ""
    expected_columns: list[str] = field(default_factory=list)
    expected_row_range: tuple = (0, 999999)
    execution_success: bool = False
    column_match: bool = False
    row_count_in_range: bool = False
    actual_row_count: int = 0
    actual_columns: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    retries_needed: int = 0
    error_message: str = ""
    # Semantic judge fields (populated only when --semantic is used)
    semantic_correct: bool = False
    semantic_score: float = 0.0
    semantic_reason: str = ""


_JUDGE_SYSTEM = """You are a strict SQL result evaluator for a Spotify music analytics app.

Given a natural language question, the SQL that was generated, and the query results,
decide if the results correctly and completely answer the question.

Scoring guide:
- 1.0 = correct answer, right columns, right ordering/direction
- 0.7 = mostly correct but minor issue (extra columns, slightly off label)
- 0.4 = partially answers the question (right topic, wrong metric or filter)
- 0.0 = wrong answer (wrong aggregation, missing WHERE, reversed ORDER, etc.)

Be strict: "highest" ordered ascending = 0.0. Missing filter = 0.0.

Respond with valid JSON only — no markdown, no explanation outside the JSON:
{"correct": true/false, "score": 0.0-1.0, "reason": "one sentence"}"""


def semantic_judge(question: str, sql: str, df: pd.DataFrame) -> dict:
    """Call LLM-as-judge to score whether the result correctly answers the question."""
    from src.llm_provider import call_llm_json

    preview = df.head(5).to_markdown(index=False)
    n = len(df)
    user = (
        f"QUESTION: {question}\n\n"
        f"SQL:\n{sql}\n\n"
        f"RESULTS ({min(5, n)} of {n} rows shown):\n{preview}"
    )
    try:
        result = call_llm_json(_JUDGE_SYSTEM, user, temperature=0.0)
        return {
            "correct": bool(result.get("correct", False)),
            "score":   min(1.0, max(0.0, float(result.get("score", 0.0)))),
            "reason":  str(result.get("reason", "")),
        }
    except Exception as e:
        logger.warning(f"Semantic judge failed: {e}")
        return {"correct": False, "score": 0.0, "reason": f"Judge error: {e}"}


def load_test_questions() -> list[dict]:
    with open(TEST_QUESTIONS_PATH) as f:
        return json.load(f)


def evaluate_question(q: dict, delay_between_calls: float = 2.0, use_semantic_judge: bool = False) -> QuestionResult:
    """Run one test question through the full pipeline."""
    from src.database import get_connection
    from src.query_executor import execute_query
    from src.sql_generator import generate_sql
    from src.sql_validator import validate_sql

    result = QuestionResult(
        id=q["id"],
        question=q["natural_language_question"],
        difficulty=q["difficulty"],
        category=q["category"],
        expected_sql=q["expected_sql"],
        expected_columns=q["expected_columns"],
        expected_row_range=(
            q["expected_row_count_range"][0],
            q["expected_row_count_range"][1],
        ),
    )

    try:
        # Generate SQL
        gen = generate_sql(q["natural_language_question"])
        result.generated_sql = gen["sql"]

        # Validate
        con = get_connection()
        validation = validate_sql(gen["sql"], con)
        if not validation.is_valid:
            result.error_message = f"Validation: {'; '.join(validation.errors)}"
            return result

        # Execute
        exec_result = execute_query(q["natural_language_question"], validation.sanitized_sql)
        result.execution_time_ms = exec_result.execution_time_ms
        result.retries_needed = exec_result.retries_needed

        if not exec_result.success:
            result.error_message = exec_result.error_message
            return result

        result.execution_success = True
        result.actual_row_count = exec_result.row_count
        result.actual_columns = list(exec_result.dataframe.columns)

        # Check row count range
        lo, hi = result.expected_row_range
        result.row_count_in_range = lo <= exec_result.row_count <= hi

        # Check column match (expected columns are a subset of actual)
        actual_lower = {c.lower() for c in result.actual_columns}
        expected_lower = {c.lower() for c in result.expected_columns}
        result.column_match = expected_lower.issubset(actual_lower)

        # Semantic correctness (LLM-as-judge)
        if use_semantic_judge and not exec_result.dataframe.empty:
            judgment = semantic_judge(
                q["natural_language_question"],
                exec_result.final_sql,
                exec_result.dataframe,
            )
            result.semantic_correct = judgment["correct"]
            result.semantic_score   = judgment["score"]
            result.semantic_reason  = judgment["reason"]

    except Exception as e:
        result.error_message = str(e)
        logger.error(f"Q{q['id']} failed: {e}")

    # Rate-limit courtesy delay
    if delay_between_calls > 0:
        time.sleep(delay_between_calls)

    return result


def run_evaluation(
    question_ids: list[int] | None = None,
    delay_between_calls: float = 2.0,
    use_semantic_judge: bool = False,
) -> list[QuestionResult]:
    """
    Run evaluation on benchmark questions.
    Pass question_ids to run a subset, or None for all 30.
    """
    questions = load_test_questions()
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]

    results = []
    total = len(questions)
    for i, q in enumerate(questions):
        logger.info(f"[{i+1}/{total}] Q{q['id']}: {q['natural_language_question'][:60]}...")
        r = evaluate_question(q, delay_between_calls=delay_between_calls,
                              use_semantic_judge=use_semantic_judge)
        results.append(r)
        status = "✅" if r.execution_success else "❌"
        sem = f" sem={r.semantic_score:.2f}" if r.semantic_score > 0 else ""
        logger.info(
            f"  {status} success={r.execution_success} rows={r.actual_row_count} "
            f"cols_ok={r.column_match} retries={r.retries_needed}{sem}"
        )

    return results


def compute_metrics(results: list[QuestionResult]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    total = len(results)
    if total == 0:
        return {}

    executed = [r for r in results if r.execution_success]
    n_executed = len(executed)

    judged = [r for r in executed if r.semantic_score > 0]

    metrics = {
        "total": total,
        "execution_success_rate": round(n_executed / total * 100, 1),
        "column_accuracy": round(sum(r.column_match for r in executed) / max(n_executed, 1) * 100, 1),
        "row_count_accuracy": round(sum(r.row_count_in_range for r in executed) / max(n_executed, 1) * 100, 1),
        "avg_execution_time_ms": round(
            sum(r.execution_time_ms for r in executed) / max(n_executed, 1), 1
        ),
        "avg_retries": round(sum(r.retries_needed for r in results) / total, 2),
        "semantic_accuracy": round(sum(r.semantic_correct for r in judged) / max(len(judged), 1) * 100, 1) if judged else None,
        "avg_semantic_score": round(sum(r.semantic_score for r in judged) / max(len(judged), 1), 2) if judged else None,
        "semantic_questions_judged": len(judged),
        "by_difficulty": {},
        "by_category": {},
    }

    for difficulty in ["easy", "medium", "hard"]:
        sub = [r for r in results if r.difficulty == difficulty]
        if sub:
            metrics["by_difficulty"][difficulty] = {
                "total": len(sub),
                "success_rate": round(sum(r.execution_success for r in sub) / len(sub) * 100, 1),
            }

    for category in ["aggregation", "filtering", "ranking", "complex"]:
        sub = [r for r in results if r.category == category]
        if sub:
            metrics["by_category"][category] = {
                "total": len(sub),
                "success_rate": round(sum(r.execution_success for r in sub) / len(sub) * 100, 1),
            }

    return metrics


def generate_report(results: list[QuestionResult], metrics: dict) -> str:
    """Generate a markdown evaluation report."""
    sem_acc = metrics.get("semantic_accuracy")
    sem_score = metrics.get("avg_semantic_score")
    sem_n = metrics.get("semantic_questions_judged", 0)

    lines = [
        "# QueryPilot Evaluation Report",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total questions | {metrics['total']} |",
        f"| SQL execution success rate | {metrics['execution_success_rate']}% |",
        f"| Column accuracy (of successful) | {metrics['column_accuracy']}% |",
        f"| Row count accuracy (of successful) | {metrics['row_count_accuracy']}% |",
        f"| **Semantic accuracy (LLM-as-judge)** | **{sem_acc}%** ({sem_n} judged) |" if sem_acc is not None else "| Semantic accuracy | not run — use `--semantic` |",
        f"| Average semantic score | {sem_score} / 1.0 |" if sem_score is not None else "",
        f"| Average execution time | {metrics['avg_execution_time_ms']}ms |",
        f"| Average retries | {metrics['avg_retries']} |",
        "",
        "## By Difficulty",
        "",
        "| Difficulty | Total | Success Rate |",
        "| --- | --- | --- |",
    ]
    for diff, stats in metrics.get("by_difficulty", {}).items():
        lines.append(f"| {diff} | {stats['total']} | {stats['success_rate']}% |")

    lines += [
        "",
        "## By Category",
        "",
        "| Category | Total | Success Rate |",
        "| --- | --- | --- |",
    ]
    for cat, stats in metrics.get("by_category", {}).items():
        lines.append(f"| {cat} | {stats['total']} | {stats['success_rate']}% |")

    lines += ["", "## Per-Question Results", "", "| ID | Difficulty | Category | Success | Rows | Cols OK | Time (ms) | Retries |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]

    for r in results:
        ok = "✅" if r.execution_success else "❌"
        cols_ok = "✅" if r.column_match else ("—" if not r.execution_success else "❌")
        rows = str(r.actual_row_count) if r.execution_success else "—"
        lines.append(
            f"| {r.id} | {r.difficulty} | {r.category} | {ok} | {rows} | {cols_ok} | {r.execution_time_ms:.0f} | {r.retries_needed} |"
        )

    # Failure analysis
    failures = [r for r in results if not r.execution_success]
    if failures:
        lines += ["", "## Failure Analysis", ""]
        for r in failures:
            lines += [
                f"### Q{r.id}: {r.question}",
                f"- **Difficulty**: {r.difficulty} | **Category**: {r.category}",
                f"- **Expected SQL**: `{r.expected_sql[:100]}...`",
                f"- **Generated SQL**: `{r.generated_sql[:100] if r.generated_sql else 'None'}...`",
                f"- **Error**: {r.error_message}",
                "",
            ]

    return "\n".join(lines)


def save_report(results: list[QuestionResult], metrics: dict) -> Path:
    """Write the evaluation report to disk."""
    report_md = generate_report(results, metrics)
    with open(REPORT_PATH, "w") as f:
        f.write(report_md)
    logger.info(f"Report saved to {REPORT_PATH}")
    return REPORT_PATH
