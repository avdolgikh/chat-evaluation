#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

from .bot_adapter import build_bot_adapter
from .runner import LangfuseTraceWriter, OfflineRunner
from .scorer import OfflineScorer, ScoredSession
from .user_simulator import Scenario, UserSimulator

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_scenarios(path: str) -> List[Scenario]:
    scenarios: List[Scenario] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            scenarios.append(
                Scenario(
                    id=data["id"],
                    persona=data["persona"],
                    goal=data["goal"],
                    constraints=list(data.get("constraints", [])),
                    max_turns=int(data["max_turns"]),
                    must_include=list(data.get("must_include", [])),
                    must_avoid=list(data.get("must_avoid", [])),
                )
            )
    return scenarios


def ensure_langfuse_env() -> Dict[str, str]:
    required = {
        "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY", ""),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            f"Missing required environment variables for Langfuse mode: {', '.join(missing)}"
        )
    return required


def run_online_contour(args, output_dir: str) -> bool:
    """
    Reuse the existing online contour for score push/dashboard compatibility.

    Fetches synthetic traces from Langfuse by user_id prefix and evaluates them with
    the online evaluator flow.
    """
    try:
        from ..online.config import config
        from ..online.main import run_evaluation
    except ImportError:
        from online.config import config
        from online.main import run_evaluation

    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = args.score_env

    config.source_environment = args.source_env
    config.score_environment = args.score_env
    config.sample_size = args.n
    config.max_per_user = args.max_per_user
    config.max_per_agent = args.max_per_agent
    config.conversation_window = args.window
    config.window_offset = max(1, args.window // 2)
    config.min_turns = args.min_turns
    config.trace_name_filter = args.trace_name
    config.user_id_prefix_filter = args.user_prefix
    config.dry_run = args.dry_run_online
    config.run_llm_metrics = True
    config.run_session_metrics = True

    end_date = datetime.now(timezone.utc) + timedelta(minutes=5)
    start_date = end_date - timedelta(hours=args.lookback_hours)
    report_path = os.path.join(output_dir, f"online_contour_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    logger.info(
        "Running online contour on synthetic traces: env=%s, user_prefix=%s, date=%s..%s",
        args.source_env,
        args.user_prefix,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    return run_evaluation(
        days_back=1,
        save_results=True,
        output_path=report_path,
        start_date=start_date,
        end_date=end_date,
    )


def save_report(output_dir: str, run_id: str, scored_sessions: List[ScoredSession], summary: Dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sessions": [
            {
                **asdict(item.session),
                "metrics": item.metrics,
            }
            for item in scored_sessions
        ],
        "summary": summary,
    }
    path = os.path.join(output_dir, f"{run_id}_report.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Evaluation Runner (User AI simulator + Langfuse-native scoring flow)"
    )
    parser.add_argument("--scenarios", default="offline/scenarios.jsonl", help="Path to scenarios JSONL")
    parser.add_argument("--n", type=int, default=20, help="Number of simulated sessions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="offline/results", help="Output directory for artifacts")
    parser.add_argument("--max-turns-override", type=int, default=None, help="Override max turns for all scenarios")

    parser.add_argument("--bot-mode", choices=["echo", "http"], default="echo", help="Bot adapter mode")
    parser.add_argument("--bot-url", default=None, help="HTTP endpoint for bot when bot-mode=http")
    parser.add_argument("--bot-timeout", type=float, default=30.0, help="HTTP timeout seconds for bot requests")

    parser.add_argument("--trace-name", default="chat_trace", help="Langfuse trace name for synthetic chats")
    parser.add_argument("--user-prefix", default="offline-eval-service", help="Prefix for synthetic user_id values")
    parser.add_argument("--agent-name", default="offline-user-ai", help="Agent name metadata for synthetic traces")
    parser.add_argument("--source-env", choices=["dev", "stg", "prod"], default="dev", help="Source env for evaluation fetch")
    parser.add_argument("--score-env", choices=["dev", "stg", "prod"], default="dev", help="Score env for evaluation push")
    parser.add_argument("--langfuse-host", default=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    parser.add_argument("--no-langfuse", action="store_true", help="Run local-only mode without Langfuse trace writes")
    parser.add_argument("--no-online-contour", action="store_true", help="Skip online contour scoring pass")
    parser.add_argument("--dry-run-online", action="store_true", help="Evaluate in online contour without pushing scores")
    parser.add_argument("--lookback-hours", type=int, default=6, help="Lookback window for online contour fetch")

    parser.add_argument("--min-turns", type=int, default=3, help="Online contour min turns")
    parser.add_argument("--window", type=int, default=10, help="Online contour conversation window")
    parser.add_argument("--max-per-user", type=int, default=1000, help="Online contour max sessions per user")
    parser.add_argument("--max-per-agent", type=int, default=1000, help="Online contour max sessions per agent")

    parser.add_argument("--use-judge-llm", action="store_true", help="Use judge LLM in offline scorer")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    scenarios_path = Path(args.scenarios)
    if not scenarios_path.exists():
        logger.error(f"Scenarios file not found: {args.scenarios}")
        return 1

    scenarios = load_scenarios(args.scenarios)
    if not scenarios:
        logger.error("No scenarios loaded")
        return 1

    headers: Dict[str, str] = {}
    bot_adapter = build_bot_adapter(
        mode=args.bot_mode,
        bot_url=args.bot_url,
        timeout_seconds=args.bot_timeout,
        headers=headers,
    )
    simulator = UserSimulator(seed=args.seed)

    trace_writer = None
    if not args.no_langfuse:
        env = ensure_langfuse_env()
        trace_writer = LangfuseTraceWriter(
            public_key=env["LANGFUSE_PUBLIC_KEY"],
            secret_key=env["LANGFUSE_SECRET_KEY"],
            host=args.langfuse_host,
            environment=args.source_env,
            trace_name=args.trace_name,
            agent_name=args.agent_name,
        )

    runner = OfflineRunner(
        bot_adapter=bot_adapter,
        user_simulator=simulator,
        trace_writer=trace_writer,
    )
    sessions = runner.run(
        scenarios=scenarios,
        n=args.n,
        seed=args.seed,
        user_prefix=args.user_prefix,
        output_dir=args.output_dir,
        max_turns_override=args.max_turns_override,
    )

    scenario_map = {scenario.id: scenario for scenario in scenarios}
    scorer = OfflineScorer(use_judge_llm=args.use_judge_llm)
    scored_sessions: List[ScoredSession] = []
    for session in sessions:
        metrics = scorer.score_session(session, scenario_map[session.scenario_id])
        scored_sessions.append(ScoredSession(session=session, metrics=metrics))

    run_id = sessions[0].run_id if sessions else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = scorer.summarize(scored_sessions)
    report_path = save_report(args.output_dir, run_id, scored_sessions, summary)

    print("=" * 60)
    print("OFFLINE EVAL SUMMARY")
    print("=" * 60)
    print(f"Sessions simulated: {len(sessions)}")
    print(f"Report: {report_path}")
    if "composite_score" in summary:
        print(f"Composite mean: {summary['composite_score']['mean']:.2f}")
    if "goal_hit" in summary:
        print(f"Goal-hit rate: {summary['goal_hit']['mean']:.2%}")

    online_ok = True
    if not args.no_langfuse and not args.no_online_contour:
        try:
            online_ok = run_online_contour(args, args.output_dir)
            print(f"Online contour run: {'SUCCESS' if online_ok else 'FAILED'}")
        except Exception as exc:
            online_ok = False
            logger.error(f"Online contour run failed: {exc}", exc_info=True)
            print("Online contour run: FAILED")

    return 0 if online_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
