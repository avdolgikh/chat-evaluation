# Chat Evaluation

A small Python package that evaluates chat conversation quality using LLM-as-a-judge. It fetches traces from [Langfuse](https://langfuse.com), scores them with a judge LLM via [Replicate](https://replicate.com), and pushes results back to Langfuse.

## How It Works

1. Fetches recent chat traces from Langfuse (grouped by session).
2. Samples sessions with diversity controls (per-user and per-agent caps).
3. Evaluates a conversation window using a judge LLM (one call per metric).
4. Pushes scores back to Langfuse and optionally saves a JSON report.
5. Computes session-level metrics (e.g., average session length) without LLM cost.

## Package Layout

```
online/
  config.py          Config + environment variables
  main.py            CLI entry point
  evaluator.py       Orchestrates fetch/eval/push/summary
  langfuse_client.py Langfuse SDK wrapper + sampling + session metrics
  judge_llm.py       Replicate judge model + prompts
  run_prod_daily.py  Day-by-day runner for large date ranges
```

## Requirements

- A Langfuse project with traces and API keys.
- A Replicate API token for the judge LLM.
- Python packages: `langfuse`, `replicate`, `python-dotenv`.

## Environment Variables

Required:
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `REPLICATE_API_TOKEN`

Optional:
- `LANGFUSE_HOST` (default: `https://cloud.langfuse.com`)
- `JUDGE_MODEL` (default in `online/config.py`)
- `REPLICATE_TIMEOUT`, `REPLICATE_USE_SYNC`
- `SOURCE_ENVIRONMENT`, `SCORE_ENVIRONMENT`

## Quick Start

```bash
# Evaluate sessions in a date range
python -m evaluation.online.main --start 2026-01-01 --end 2026-01-07 --samples 5

# Dry run (evaluate but don't push scores)
python -m evaluation.online.main --start 2026-01-01 --end 2026-01-07 --samples 5 --dry-run

# Validate environment/config only
python -m evaluation.online.main --start 2026-01-01 --validate-only
```

For large date ranges, use the day-by-day runner:

```bash
python online/run_prod_daily.py --start 2026-01-01 --end 2026-01-07
```

See `--help` for all available options (sampling controls, environment selection, metric selection, etc.).

## Scoring

LLM metrics (1–5 each): **relevance**, **engagement**, **naturalness**, **appropriateness**.

Composite score = sum of the four LLM metrics (range 4–20).

Session metrics (no LLM cost): **avg_session_length**.
