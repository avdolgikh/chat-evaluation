# Online Evaluation Specification (Basic)

## Purpose

Run daily chat quality evaluation on real production/staging/dev traces, score sampled sessions, and publish metrics.

## Inputs

- Date range: `start` (required), `end` (optional).
- Source environment: where traces are fetched (`dev|stg|prod`).
- Score environment: where scores are written (`dev|stg|prod`).
- Sampling controls:
  - sessions per day
  - max sessions per user
  - max sessions per agent
- Conversation controls:
  - minimum turns per session
  - evaluation window size

## Required Secrets

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `REPLICATE_API_TOKEN`

## Evaluation Flow

1. Fetch traces from Langfuse for the date range and source environment.
2. Keep chat sessions only and group by `session_id`.
3. Compute session-level metric on all sessions: `avg_session_length`.
4. Filter sessions by `min_turns`.
5. Sample sessions with diversity constraints (per user/per agent) per day.
6. For each sampled session, evaluate a conversation window using judge LLM.
7. Score metrics:
   - `relevance`
   - `engagement`
   - `naturalness`
   - `appropriateness`
   - `composite_score` (sum of the 4 metrics)
8. Push scores back to Langfuse (unless `dry_run=true`).
9. Print summary and optionally save JSON results locally.

## CLI (minimum)

```bash
python -m evaluation.online.main --start 2026-01-01 --end 2026-01-07 --samples 100
```

Optional:

- `--dry-run`
- `--validate-only`
- `--source-env` / `--score-env`
- `--window`, `--min-turns`

## Outputs

- Per-run console summary (counts, success rate, metric means/min/max).
- Optional JSON report with per-session scores.
- Scores attached to traces in Langfuse.

## MVP Acceptance Criteria

- Run completes for a valid date range with configured secrets.
- LLM metrics are produced for sampled sessions.
- `avg_session_length` is produced from all sessions.
- Dry-run mode performs evaluation without writing scores.
