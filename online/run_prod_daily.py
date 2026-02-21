#!/usr/bin/env python
"""
Run evaluation day-by-day for a date range.

Usage:
    python run_prod_daily.py --start 2026-01-01 --end 2026-01-07
    python run_prod_daily.py --metrics avg_session_length
    python run_prod_daily.py --metrics relevance,engagement
"""
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone

# Set environment variables BEFORE any imports
os.environ["SOURCE_ENVIRONMENT"] = "prod"
os.environ["SCORE_ENVIRONMENT"] = "prod"
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "prod"  # SDK uses this for score env

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from evaluation.online.config import config
from evaluation.online.main import run_evaluation

# Configuration
START_DATE = "2026-01-01"
END_DATE = "2026-01-07"
PAGES_PER_DAY = 100
SAMPLES_PER_DAY = 20

def parse_metrics_arg(metrics_str: str) -> tuple:
    """
    Parse --metrics argument.

    Returns: (run_llm_metrics, run_session_metrics)

    Examples:
        "all" or None -> (True, True)  # Default: all 6 metrics
        "llm" -> (True, False)  # Only LLM metrics
        "avg_session_length" or "session" -> (False, True)  # Only session metrics
        "relevance,engagement" -> (True, False) + sets config.llm_metrics
    """
    if not metrics_str or metrics_str.lower() == "all":
        return True, True

    metrics_str = metrics_str.lower().strip()

    if metrics_str == "llm":
        return True, False

    if metrics_str in ("avg_session_length", "session", "session_metrics"):
        return False, True

    # Custom list of LLM metrics
    custom_metrics = [m.strip() for m in metrics_str.split(",")]
    valid_llm = ["relevance", "engagement", "naturalness", "appropriateness"]

    # Filter to valid LLM metrics
    selected_llm = [m for m in custom_metrics if m in valid_llm]
    has_session = "avg_session_length" in custom_metrics

    if selected_llm:
        config.llm_metrics = selected_llm

    return bool(selected_llm), has_session

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PROD evaluation day-by-day")
    parser.add_argument("--metrics", type=str, default="all",
                       help="Metrics to run: 'all' (default), 'llm', 'avg_session_length', or comma-separated list")
    parser.add_argument("--start", type=str, default=START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=END_DATE, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Parse metrics
    run_llm, run_session = parse_metrics_arg(args.metrics)
    config.run_llm_metrics = run_llm
    config.run_session_metrics = run_session

    print("=" * 60)
    print("PROD EVALUATION: Day-by-Day")
    print("=" * 60)
    print(f"Date range: {args.start} to {args.end}")
    print(f"Pages per day: {PAGES_PER_DAY}")
    print(f"Samples per day: {SAMPLES_PER_DAY}")
    print(f"Metrics: {args.metrics}")
    print(f"  Run LLM metrics: {run_llm} ({config.llm_metrics if run_llm else 'none'})")
    print(f"  Run session metrics: {run_session} (avg_session_length)")
    print("=" * 60)
    print()

    # Configure settings
    config.source_environment = "prod"
    config.score_environment = "prod"
    config.min_turns = 3
    config.conversation_window = 10
    config.window_offset = 5
    config.sample_size = SAMPLES_PER_DAY
    config.max_session_pages = PAGES_PER_DAY

    # Parse dates from args
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    # Track results
    day_results = []

    # Loop through each day
    current = start
    while current <= end:
        day_str = current.strftime("%Y-%m-%d")
        next_day = current + timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")

        print()
        print("=" * 60)
        print(f"Processing: {day_str}")
        print("=" * 60)

        try:
            # Call run_evaluation directly (doesn't call sys.exit)
            # Note: pass datetime objects, not strings
            from datetime import timezone as tz
            start_dt = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=tz.utc)
            end_dt = datetime.strptime(next_day_str, "%Y-%m-%d").replace(tzinfo=tz.utc)

            success = run_evaluation(
                days_back=None,
                save_results=True,
                output_path=None,
                start_date=start_dt,
                end_date=end_dt
            )
            day_results.append((day_str, "SUCCESS" if success else "FAILED"))
        except Exception as e:
            print(f"Error processing {day_str}: {e}")
            import traceback
            traceback.print_exc()
            day_results.append((day_str, f"ERROR: {e}"))

        current = next_day

    print()
    print("=" * 60)
    print("Day-by-day evaluation complete!")
    print("=" * 60)
    print()
    print("Summary:")
    for day, result in day_results:
        print(f"  {day}: {result}")
