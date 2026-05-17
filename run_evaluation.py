"""
Evaluation runner: execute all 30 benchmark questions and generate a report.

Usage:
    python run_evaluation.py                  # all 30 questions
    python run_evaluation.py --ids 1 2 3      # specific questions
    python run_evaluation.py --difficulty easy # filter by difficulty
    python run_evaluation.py --fast           # no delay (use only on paid APIs)
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Run QueryPilot evaluation benchmark")
    parser.add_argument("--ids", nargs="+", type=int, help="Specific question IDs to run")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty level",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip rate-limit delays (use only with paid API tiers)",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable LLM-as-judge semantic scoring (doubles API calls)",
    )
    args = parser.parse_args()

    # Ensure database exists
    from src.database import DB_PATH, setup_database
    if not DB_PATH.exists():
        logging.info("Database not found — building it now...")
        setup_database()

    from src.evaluation import (
        compute_metrics,
        load_test_questions,
        run_evaluation,
        save_report,
    )

    # Determine which questions to run
    question_ids = args.ids
    if args.difficulty and not question_ids:
        questions = load_test_questions()
        question_ids = [q["id"] for q in questions if q["difficulty"] == args.difficulty]
        logging.info(f"Running {len(question_ids)} '{args.difficulty}' questions")

    delay = 0.5 if args.fast else 2.0
    logging.info(f"Delay between calls: {delay}s")
    if args.semantic:
        logging.info("Semantic judge enabled (LLM-as-judge scoring)")

    results = run_evaluation(
        question_ids=question_ids,
        delay_between_calls=delay,
        use_semantic_judge=args.semantic,
    )
    metrics = compute_metrics(results)

    # Print summary
    print("\n" + "=" * 55)
    print("  QueryPilot Evaluation Results")
    print("=" * 55)
    print(f"  Total questions:        {metrics['total']}")
    print(f"  Execution success rate: {metrics['execution_success_rate']}%")
    print(f"  Column accuracy:        {metrics['column_accuracy']}%")
    print(f"  Row count accuracy:     {metrics['row_count_accuracy']}%")
    print(f"  Avg execution time:     {metrics['avg_execution_time_ms']}ms")
    print(f"  Avg retries:            {metrics['avg_retries']}")
    if metrics.get("semantic_accuracy") is not None:
        print(f"  Semantic accuracy:      {metrics['semantic_accuracy']}%  ({metrics['semantic_questions_judged']} judged)")
        print(f"  Avg semantic score:     {metrics['avg_semantic_score']} / 1.0")
    print()
    print("  By Difficulty:")
    for diff, stats in metrics.get("by_difficulty", {}).items():
        print(f"    {diff:8s}: {stats['success_rate']:5.1f}% ({stats['total']} questions)")
    print()
    print("  By Category:")
    for cat, stats in metrics.get("by_category", {}).items():
        print(f"    {cat:15s}: {stats['success_rate']:5.1f}% ({stats['total']} questions)")
    print("=" * 55)

    report_path = save_report(results, metrics)
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
