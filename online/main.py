#!/usr/bin/env python3
"""
Online Evaluation System for Langfuse Traces

This script runs the daily evaluation process for chat conversations:
1. Fetches recent chat traces from Langfuse (grouped by session)
2. Evaluates conversation sessions using a judge LLM
3. Pushes scores back to Langfuse
4. Generates summary statistics
5. Logs detailed scoring for each session
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Optional
from .evaluator import OnlineEvaluator
from .config import config

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'conversation_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def validate_environment() -> bool:
    """Validate that all required environment variables are set"""
    required_vars = [
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PUBLIC_KEY", 
        "REPLICATE_API_TOKEN"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the evaluation.")
        return False
    
    return True

def run_evaluation(days_back: int = 1,
                  save_results: bool = True,
                  output_path: Optional[str] = None,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> bool:
    """
    Run the evaluation process for chat conversations

    Args:
        days_back: Number of days back to fetch traces from
        save_results: Whether to save results to file
        output_path: Path to save results (optional)
        start_date: Explicit start date for filtering
        end_date: Explicit end date for filtering

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize evaluator
        evaluator = OnlineEvaluator()

        # Run evaluation
        results = evaluator.run_daily_evaluation(
            days_back=days_back,
            start_date=start_date,
            end_date=end_date
        )

        # Check if session metrics were pushed (even if no LLM results)
        session_metrics_pushed = evaluator.get_abandonment_metrics() is not None

        if not results:
            if not config.run_llm_metrics and session_metrics_pushed:
                # Only session metrics were requested and pushed successfully
                print("Session metrics (avg_session_length) pushed successfully")
                return True
            print("No conversation evaluation results generated")
            return False
        
        # Generate summary
        summary = evaluator.get_evaluation_summary(results)
        
        # Print summary
        print("\n" + "="*60)
        print("CONVERSATION EVALUATION SUMMARY")
        print("="*60)
        print(f"Total conversation sessions: {summary.get('total_evaluations', 0)}")
        print(f"Successfully evaluated sessions: {summary.get('successful_evaluations', 0)}")
        print(f"Success rate: {summary.get('success_rate', 0):.2%}")
        
        if 'composite_score' in summary:
            comp_score = summary['composite_score']
            print(f"Composite score - Mean: {comp_score.get('mean', 0):.2f}, "
                  f"Min: {comp_score.get('min', 0):.2f}, "
                  f"Max: {comp_score.get('max', 0):.2f}")
        
        # Print metric scores
        if 'metric_scores' in summary:
            print("\nMetric Scores (per conversation session):")
            for metric, stats in summary['metric_scores'].items():
                print(f"  {metric}: {stats.get('mean', 0):.2f} "
                      f"(min: {stats.get('min', 0):.2f}, "
                      f"max: {stats.get('max', 0):.2f})")

        # Print session metrics (100% coverage, no Judge LLM)
        if 'abandonment_metrics' in summary:
            am = summary['abandonment_metrics']
            print("\n" + "-"*60)
            print("SESSION METRICS (100% coverage, no Judge LLM)")
            print("-"*60)
            print(f"  Total sessions analyzed: {am.get('total_sessions', 0)}")
            print(f"  Avg session length: {am.get('avg_session_length', 0):.1f} turns")

        # Save results if requested
        if save_results:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"conversation_evaluation_results_{timestamp}.json"
            
            success = evaluator.save_evaluation_results(results, output_path)
            if success:
                print(f"\nResults saved to: {output_path}")
            else:
                print("\nWarning: Failed to save results to file")
        
        return True
        
    except Exception as e:
        print(f"Error running conversation evaluation: {e}")
        logging.error(f"Conversation evaluation failed: {e}", exc_info=True)
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Online Evaluation System for Chat Conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.online.main --start 2026-01-01 --end 2026-01-07 --samples 5
  python -m evaluation.online.main --start 2026-01-01 --end 2026-01-07 --samples 5 --dry-run
        """
    )

    # === Date Range Parameters ===
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date for evaluation (YYYY-MM-DD format, required)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date for evaluation (YYYY-MM-DD format, default: today)"
    )

    # === Sampling Parameters ===
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of sessions to evaluate (default: 100)"
    )
    parser.add_argument(
        "--max-per-user",
        type=int,
        default=5,
        help="Max sessions per user_id for diversity (default: 5)"
    )
    parser.add_argument(
        "--max-per-agent",
        type=int,
        default=10,
        help="Max sessions per agent_id for diversity (default: 10)"
    )

    # === Conversation Window Parameters ===
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Number of turns to evaluate per session (default: 10)"
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=10,
        help="Minimum turns required in session (default: 10)"
    )

    # === Environment Parameters ===
    parser.add_argument(
        "--source-env",
        type=str,
        choices=["dev", "stg", "prod"],
        default="prod",
        help="Source environment to fetch traces from (default: prod)"
    )
    parser.add_argument(
        "--score-env",
        type=str,
        choices=["dev", "stg", "prod"],
        default="dev",
        help="Environment to push scores to (default: dev)"
    )

    # === Output Parameters ===
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for results"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    # === Utility Parameters ===
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate environment and configuration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate but don't push scores to Langfuse"
    )

    args = parser.parse_args()

    # Apply CLI args to config
    config.source_environment = args.source_env
    config.score_environment = args.score_env
    config.sample_size = args.samples
    config.max_per_user = args.max_per_user
    config.max_per_agent = args.max_per_agent
    config.conversation_window = args.window
    config.min_turns = args.min_turns
    config.dry_run = args.dry_run

    # CRITICAL: Set LANGFUSE_TRACING_ENVIRONMENT for Langfuse SDK
    # This must be set BEFORE Langfuse client is created (in run_evaluation)
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = args.score_env

    # Parse date arguments
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
        days_back = (datetime.now() - start_date).days
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD. {e}")
        sys.exit(1)

    # Setup logging
    setup_logging(args.log_level)

    print("Online Evaluation System for Chat Conversations")
    print("="*60)

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Print configuration
    print(f"Configuration:")
    print(f"  Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days_back} days)")
    print(f"  Source Environment: {config.source_environment} (fetch traces)")
    print(f"  Score Environment: {config.score_environment} (push scores)")
    print(f"  Dry Run: {config.dry_run}")
    print(f"Sampling:")
    print(f"  Samples: {config.sample_size} sessions/day")
    print(f"  Max per User: {config.max_per_user} (diversity)")
    print(f"  Max per agent: {config.max_per_agent} (diversity)")
    print(f"Conversation Window:")
    print(f"  Window Size: {config.conversation_window} turns")
    print(f"  Min Turns Required: {config.min_turns}")
    print(f"Judge LLM:")
    print(f"  Model: {config.judge_model.split(':')[0]}")
    print(f"  Timeout: {config.replicate_timeout}s")
    print()

    if args.validate_only:
        print("Environment validation passed. Configuration looks good.")
        return

    # Run evaluation
    success = run_evaluation(
        days_back=days_back,
        save_results=not args.no_save,
        output_path=args.output,
        start_date=start_date,
        end_date=end_date
    )
    
    if success:
        print("\nConversation evaluation completed successfully!")
        sys.exit(0)
    else:
        print("\nConversation evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
