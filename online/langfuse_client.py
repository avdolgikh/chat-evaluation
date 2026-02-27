import logging
import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langfuse import Langfuse
from .config import config

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None
    span_id: Optional[str] = None

@dataclass
class ConversationTrace:
    """Represents a conversation trace from Langfuse"""
    trace_id: str  # Last trace ID in session (for score association)
    user_id: Optional[str]
    session_id: Optional[str]
    conversation: List[ConversationTurn]
    metadata: Dict[str, Any]
    created_at: datetime
    total_turns: int
    last_trace_timestamp: Optional[datetime] = None  # For score timestamp (dashboards)

# Valid types for evaluation
VALID_TYPES = ['chat']


@dataclass
class AbandonmentMetrics:
    """
    Session abandonment metrics - calculated WITHOUT Judge LLM.

    These metrics provide 100% session coverage at zero LLM cost.
    They measure user engagement/churn signals that complement
    Judge LLM quality scores (which only cover sampled sessions).

    Use Cases:
        - Track user churn over time (abandonment_rate trend)
        - Detect quality regressions (rising short_session_rate)
        - Compare engagement across environments
        - Correlate with Judge LLM scores (do low-quality = high abandonment?)

    Thresholds:
        - Abandoned: < 3 turns (user left very early)
        - Short: < 5 turns (minimal engagement)
        - Engaged: >= 10 turns (meaningful conversation)

    Attributes:
        total_sessions: Number of sessions analyzed (100% of sessions in date range)
        abandonment_rate: Fraction of sessions with < 3 turns (0.0 to 1.0)
        short_session_rate: Fraction of sessions with < 5 turns (0.0 to 1.0)
        engaged_session_rate: Fraction of sessions with >= 10 turns (0.0 to 1.0)
        avg_session_length: Mean turns per session (sensitive to outliers)
        median_session_length: Median turns (p50, more robust than mean)
        min_turns: Shortest session in date range
        max_turns: Longest session in date range
        date_range_start: Start of evaluation date range (for reporting)
        date_range_end: End of evaluation date range (for reporting)

    Example:
        >>> metrics = calculate_abandonment_metrics([2, 3, 5, 10, 15, 20])
        >>> metrics.abandonment_rate  # 1/6 = 0.167 (one session < 3 turns)
        >>> metrics.engaged_session_rate  # 3/6 = 0.5 (three sessions >= 10 turns)
    """
    total_sessions: int
    abandonment_rate: float
    short_session_rate: float
    engaged_session_rate: float
    avg_session_length: float
    median_session_length: float
    min_turns: int
    max_turns: int
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None


def calculate_abandonment_metrics(
    session_turn_counts: List[int],
    date_range_start: Optional[datetime] = None,
    date_range_end: Optional[datetime] = None
) -> AbandonmentMetrics:
    """
    Calculate session abandonment metrics from turn counts.

    These metrics require NO Judge LLM - purely statistical.
    Should be calculated on ALL sessions (100% coverage).

    Args:
        session_turn_counts: List of turn counts per session
        date_range_start: Start of date range (for reporting)
        date_range_end: End of date range (for reporting)

    Returns:
        AbandonmentMetrics with all calculated values
    """
    if not session_turn_counts:
        return AbandonmentMetrics(
            total_sessions=0,
            abandonment_rate=0.0,
            short_session_rate=0.0,
            engaged_session_rate=0.0,
            avg_session_length=0.0,
            median_session_length=0.0,
            min_turns=0,
            max_turns=0,
            date_range_start=date_range_start,
            date_range_end=date_range_end
        )

    total = len(session_turn_counts)
    sorted_counts = sorted(session_turn_counts)

    # Calculate rates
    abandoned = sum(1 for tc in session_turn_counts if tc < 3)
    short = sum(1 for tc in session_turn_counts if tc < 5)
    engaged = sum(1 for tc in session_turn_counts if tc >= 10)

    # Median (p50)
    mid_idx = total // 2
    if total % 2 == 0:
        median = (sorted_counts[mid_idx - 1] + sorted_counts[mid_idx]) / 2
    else:
        median = sorted_counts[mid_idx]

    return AbandonmentMetrics(
        total_sessions=total,
        abandonment_rate=abandoned / total,
        short_session_rate=short / total,
        engaged_session_rate=engaged / total,
        avg_session_length=sum(session_turn_counts) / total,
        median_session_length=median,
        min_turns=min(session_turn_counts),
        max_turns=max(session_turn_counts),
        date_range_start=date_range_start,
        date_range_end=date_range_end
    )


def sample_with_diversity(
    conversation_traces: List[ConversationTrace],
    target_samples: int,
    max_per_user: int,
    max_per_agent: int
) -> List[ConversationTrace]:
    """
    Sample conversation traces with diversity controls.

    Ensures no single user or agent dominates the evaluation by limiting
    the number of sessions per user_id and agent_id.

    Args:
        conversation_traces: List of all valid conversation traces
        target_samples: Target number of samples to return
        max_per_user: Maximum sessions per user_id
        max_per_agent: Maximum sessions per agent_id

    Returns:
        List of sampled conversation traces with diversity applied
    """
    if not conversation_traces:
        return []

    # Shuffle to randomize before applying diversity filters
    shuffled = conversation_traces.copy()
    random.shuffle(shuffled)

    user_counts = defaultdict(int)
    agent_counts = defaultdict(int)
    eligible = []

    for trace in shuffled:
        # Safe user_id extraction
        try:
            user_id = trace.user_id or 'unknown_user'
        except Exception:
            user_id = 'unknown_user'

        # Safe agent_id extraction from multiple sources:
        # 1. metadata.agent_name
        # 2. metadata.agent_id
        # 3. session_id format: {user_id}_{agent_id}
        agent_id = None
        try:
            if trace.metadata and isinstance(trace.metadata, dict):
                agent_id = trace.metadata.get('agent_name') or trace.metadata.get('agent_id')
            if not agent_id:
                session_id = getattr(trace, 'session_id', None)
                if session_id and isinstance(session_id, str) and '_' in session_id:
                    agent_id = session_id.split('_')[-1]
        except Exception:
            pass  # Fall through to default
        if not agent_id:
            agent_id = 'unknown'

        # Check diversity limits
        if user_counts[user_id] >= max_per_user:
            logger.debug(f"Skipping session for user {user_id[:8]}... (max {max_per_user} reached)")
            continue

        if agent_counts[agent_id] >= max_per_agent:
            logger.debug(f"Skipping session for agent {agent_id[:8]}... (max {max_per_agent} reached)")
            continue

        # Accept this session
        eligible.append(trace)
        user_counts[user_id] += 1
        agent_counts[agent_id] += 1

        # Stop if we have enough samples
        if len(eligible) >= target_samples:
            break

    # Log diversity stats
    logger.info(f"Diversity sampling: {len(conversation_traces)} total -> {len(eligible)} selected")
    logger.info(f"  Unique users: {len(user_counts)}, unique agents: {len(agent_counts)}")

    return eligible


def sample_per_day_with_diversity(
    conversation_traces: List[ConversationTrace],
    samples_per_day: int,
    max_per_user: int,
    max_per_agent: int
) -> List[ConversationTrace]:
    """
    Sample conversation traces with per-day quotas and diversity controls.

    Groups traces by date (using last_trace_timestamp), then samples
    `samples_per_day` sessions from each day with diversity controls.

    Args:
        conversation_traces: List of all valid conversation traces
        samples_per_day: Target number of samples PER DAY
        max_per_user: Maximum sessions per user_id (diversity)
        max_per_agent: Maximum sessions per agent_id (diversity)

    Returns:
        List of sampled conversation traces across all days
    """
    if not conversation_traces:
        return []

    # Group traces by date (using last_trace_timestamp)
    traces_by_date: Dict[str, List[ConversationTrace]] = defaultdict(list)
    for trace in conversation_traces:
        if trace.last_trace_timestamp:
            date_key = trace.last_trace_timestamp.strftime("%Y-%m-%d")
        else:
            date_key = "unknown"
        traces_by_date[date_key].append(trace)

    # Sample from each day
    all_sampled = []
    sorted_dates = sorted(traces_by_date.keys())

    for date_key in sorted_dates:
        day_traces = traces_by_date[date_key]
        day_sampled = sample_with_diversity(
            conversation_traces=day_traces,
            target_samples=samples_per_day,
            max_per_user=max_per_user,
            max_per_agent=max_per_agent
        )
        logger.info(f"  {date_key}: {len(day_traces)} eligible -> {len(day_sampled)} sampled")
        all_sampled.extend(day_sampled)

    logger.info(f"Per-day sampling complete: {len(all_sampled)} total across {len(sorted_dates)} days")
    return all_sampled


@dataclass
class TracesFetchResult:
    """Result from fetching traces - includes both eligible traces and ALL session turn counts"""
    eligible_traces: List[ConversationTrace]  # Traces that pass min_turns filter
    all_session_turn_counts: List[int]        # Turn counts for ALL sessions (for abandonment metrics)
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None


class LangfuseClient:
    """Client for interacting with Langfuse API using the official SDK"""

    def __init__(self):
        self.secret_key = config.langfuse_secret_key
        self.public_key = config.langfuse_public_key
        self.host = config.langfuse_host

        if not self.secret_key or not self.public_key:
            raise ValueError("Langfuse credentials not configured")

        # Initialize Langfuse client using the SDK with longer timeout
        self.langfuse = Langfuse(
            public_key=self.public_key,
            secret_key=self.secret_key,
            host=self.host,
            timeout=60  # 60 second timeout for API calls
        )

        # Track turn counts from last fetch (for abandonment metrics)
        self._last_fetch_turn_counts: List[int] = []
        self._last_fetch_date_range: Optional[tuple] = None

    def get_abandonment_metrics(self) -> Optional[AbandonmentMetrics]:
        """
        Get abandonment metrics from the last trace fetch.

        Must be called AFTER get_traces_for_evaluation().
        Returns None if no fetch has been done.

        Returns:
            AbandonmentMetrics calculated from ALL sessions (100% coverage)
        """
        if not self._last_fetch_turn_counts:
            logger.warning("No turn counts available - call get_traces_for_evaluation() first")
            return None

        date_start, date_end = self._last_fetch_date_range or (None, None)
        return calculate_abandonment_metrics(
            session_turn_counts=self._last_fetch_turn_counts,
            date_range_start=date_start,
            date_range_end=date_end
        )

    def _log_abandonment_metrics(self, metrics: AbandonmentMetrics) -> None:
        """Log abandonment metrics to console."""
        date_range = ""
        if metrics.date_range_start and metrics.date_range_end:
            date_range = f"{metrics.date_range_start.strftime('%Y-%m-%d')} to {metrics.date_range_end.strftime('%Y-%m-%d')}"

        logger.info(f"=== SESSION METRICS ({date_range}) ===")
        logger.info(f"  Total sessions: {metrics.total_sessions}")
        logger.info(f"  Avg session length: {metrics.avg_session_length:.1f} turns")

    def push_abandonment_metrics(self, metrics: AbandonmentMetrics) -> bool:
        """
        Push abandonment metrics to Langfuse as aggregate scores.

        Creates a synthetic "evaluation_run" trace to attach scores to.
        This allows abandonment metrics to appear in Langfuse dashboards.

        Args:
            metrics: AbandonmentMetrics to push

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check dry_run mode
            if config.dry_run:
                logger.info("[DRY RUN] Would push abandonment metrics to Langfuse")
                # Still log metrics for visibility
                self._log_abandonment_metrics(metrics)
                return True

            # Create a comment with context
            date_range = ""
            if metrics.date_range_start and metrics.date_range_end:
                date_range = f"{metrics.date_range_start.strftime('%Y-%m-%d')} to {metrics.date_range_end.strftime('%Y-%m-%d')}"

            # Create trace using SDK observation system (trace_id auto-generated)
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="session_metrics",
                metadata={
                    "type": "aggregate_metrics",
                    "date_range": date_range,
                    "total_sessions": metrics.total_sessions,
                    "source_environment": config.source_environment,
                },
            ) as span:
                span.update(
                    input={"type": "aggregate_metrics", "date_range": date_range},
                    output=f"avg_session_length={metrics.avg_session_length:.2f}",
                )
                # Get auto-generated trace_id while inside context
                trace_id = self.langfuse.get_current_trace_id()

            # Flush to ensure trace is created before attaching scores
            self.langfuse.flush()

            metric_values = {
                "avg_session_length": metrics.avg_session_length,
            }

            comment = f"Aggregate | Date: {date_range} | Sessions: {metrics.total_sessions} | source={config.source_environment}"

            for metric_name, value in metric_values.items():
                score_kwargs = {
                    "trace_id": trace_id,
                    "name": metric_name,
                    "value": value,
                    "comment": comment,
                }
                # Add timestamp for time-series dashboards (use date being evaluated)
                if metrics.date_range_start:
                    score_kwargs["timestamp"] = metrics.date_range_start
                self.langfuse.create_score(**score_kwargs)

            # Log summary
            self._log_abandonment_metrics(metrics)
            logger.info(f"Successfully pushed abandonment metrics with trace_id={trace_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to push abandonment metrics: {e}")
            return False

    def get_traces_for_evaluation(self, days_back: int = 1,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> List[ConversationTrace]:
        """
        Fetch traces from Langfuse for evaluation using the SDK
        Fetch traces directly with date filter

        Args:
            days_back: Number of days back to fetch traces from (used if start/end not provided)
            start_date: Explicit start date for filtering
            end_date: Explicit end date for filtering

        Returns:
            List of conversation traces suitable for evaluation (grouped by session)
        """
        # Calculate date range
        from datetime import timezone

        if start_date is None or end_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

        # Ensure both dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        logger.info(f"Fetching traces: env={config.source_environment}, date={start_date} to {end_date}")

        try:
            # Fetch traces directly with date filtering (more efficient than sessions-first approach)
            all_traces = []
            page = 1
            max_pages = config.max_session_pages

            while page <= max_pages:
                logger.info(f"Fetching traces page {page}...")
                try:
                    trace_params = {
                        "page": page,  # CRITICAL: page param for pagination!
                        "limit": 100,
                        "from_timestamp": start_date,
                        "to_timestamp": end_date
                    }
                    # Add name filter if configured
                    if config.trace_name_filter:
                        trace_params["name"] = config.trace_name_filter
                    # Add environment filter at API level
                    if config.source_environment:
                        trace_params["environment"] = [config.source_environment]

                    traces_response = self.langfuse.api.trace.list(**trace_params)
                    traces = traces_response.data

                    if not traces:
                        logger.info(f"No more traces found on page {page}")
                        break

                    logger.info(f"Retrieved {len(traces)} traces from page {page}")
                    all_traces.extend(traces)

                    if len(traces) < 100:
                        break
                    page += 1

                except Exception as e:
                    logger.error(f"Failed to fetch traces page {page}: {e}")
                    break

            logger.info(f"Retrieved {len(all_traces)} total traces from Langfuse")

            # Filter traces by type and group by session
            session_traces_map = defaultdict(list)
            skipped_env = 0
            skipped_type = 0
            skipped_user = 0
            type_counts = defaultdict(int)
            env_counts = defaultdict(int)
            for trace in all_traces:
                trace_metadata = getattr(trace, 'metadata', {}) or {}
                trace_type = trace_metadata.get('mode', '')
                # Environment is a top-level field on trace object (NOT in metadata)
                trace_env = getattr(trace, 'environment', '') or ''
                trace_user_id = getattr(trace, 'user_id', None) or ''
                session_id = getattr(trace, 'session_id', None)

                # Track stats for debugging
                type_counts[trace_type or 'empty'] += 1
                env_counts[trace_env or 'empty'] += 1

                # Skip non-chat types
                if trace_type not in VALID_TYPES:
                    skipped_type += 1
                    continue

                # Skip traces that don't match source environment (if trace has env set)
                if config.source_environment and trace_env and trace_env != config.source_environment:
                    skipped_env += 1
                    continue

                # Optional user_id prefix filter (used by offline eval traffic isolation)
                if config.user_id_prefix_filter:
                    if not trace_user_id.startswith(config.user_id_prefix_filter):
                        skipped_user += 1
                        continue

                if session_id:
                    session_traces_map[session_id].append(trace)

            # Log type and env distributions
            logger.info(f"Trace types: {dict(type_counts)}")
            logger.info(f"Trace envs: {dict(env_counts)}")
            if skipped_type > 0:
                logger.info(f"Skipped {skipped_type} traces with non-valid type (valid: {VALID_TYPES})")
            if skipped_env > 0:
                logger.info(f"Skipped {skipped_env} traces with non-matching environment (looking for: {config.source_environment})")
            if skipped_user > 0:
                logger.info(f"Skipped {skipped_user} traces with non-matching user_id prefix (looking for: {config.user_id_prefix_filter})")

            logger.info(f"Found {len(session_traces_map)} sessions with chat traces")

            # Process each session into a conversation trace
            # Track ALL turn counts for abandonment metrics (100% coverage)
            all_session_turn_counts = []
            all_valid_traces = []

            for session_id, session_traces in session_traces_map.items():
                try:
                    # Process session into conversation
                    conv_trace = self._process_session_to_conversation(session_id, session_traces)
                    if conv_trace:
                        # Track ALL turn counts (for abandonment metrics)
                        all_session_turn_counts.append(conv_trace.total_turns)

                        # Only include in evaluation if passes min_turns threshold
                        if conv_trace.total_turns >= config.min_turns:
                            all_valid_traces.append(conv_trace)
                            logger.debug(f"Session {session_id}: {conv_trace.total_turns} turns (eligible)")
                        else:
                            logger.debug(f"Session {session_id}: {conv_trace.total_turns} turns (below min_turns={config.min_turns})")

                except Exception as e:
                    logger.warning(f"Failed to process session {session_id}: {e}")
                    continue

            logger.info(f"Session stats: {len(all_session_turn_counts)} total, {len(all_valid_traces)} eligible (>= {config.min_turns} turns)")

            # Store turn counts for abandonment metrics (accessible via get_last_fetch_turn_counts)
            self._last_fetch_turn_counts = all_session_turn_counts
            self._last_fetch_date_range = (start_date, end_date)

            # Apply per-day diversity sampling (sample_size = sessions PER DAY)
            sampled_traces = sample_per_day_with_diversity(
                conversation_traces=all_valid_traces,
                samples_per_day=config.sample_size,
                max_per_user=config.max_per_user,
                max_per_agent=config.max_per_agent
            )

            return sampled_traces

        except Exception as e:
            logger.error(f"Failed to fetch traces: {e}")
            return []
    
    def _get_traces_for_session(self, session_id: str,
                                from_timestamp: Optional[datetime] = None,
                                to_timestamp: Optional[datetime] = None) -> List:
        """
        Get all traces for a specific session

        Args:
            session_id: The session ID to get traces for
            from_timestamp: Optional start date for filtering
            to_timestamp: Optional end date for filtering

        Returns:
            List of traces for the session
        """
        try:
            all_traces = []
            page = 1
            max_pages = 5  # Limit pages per session to avoid infinite loops

            # Implement pagination for session traces
            while page <= max_pages:
                try:
                    trace_params = {
                        "session_id": session_id,
                        "limit": 100  # Langfuse API limit
                    }
                    # Add date filtering if provided
                    if from_timestamp:
                        trace_params["from_timestamp"] = from_timestamp
                    if to_timestamp:
                        trace_params["to_timestamp"] = to_timestamp

                    traces_response = self.langfuse.api.trace.list(**trace_params)
                    traces = traces_response.data
                    
                    if not traces:
                        logger.debug(f"No traces found for session {session_id} on page {page}")
                        break
                    
                    all_traces.extend(traces)
                    logger.debug(f"Retrieved {len(traces)} traces for session {session_id} on page {page}")
                    
                    # Check if this is the last page
                    if len(traces) < trace_params["limit"]:
                        logger.debug(f"Reached last page for session {session_id}")
                        break
                    
                    page += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch traces page {page} for session {session_id}: {e}")
                    break
            
            logger.debug(f"Retrieved {len(all_traces)} total traces for session {session_id}")
            return all_traces
            
        except Exception as e:
            logger.error(f"Failed to get traces for session {session_id}: {e}")
            return []
    
    def _process_session_to_conversation(self, session_id: str, session_traces: List) -> Optional[ConversationTrace]:
        """
        Process a session (conversation) from multiple traces into a single conversation

        Args:
            session_id: The session ID representing the conversation
            session_traces: List of traces that belong to this session

        Returns:
            ConversationTrace representing the full conversation
        """
        try:
            if not session_traces:
                logger.warning(f"Session {session_id} has no traces")
                return None

            # Sort traces by timestamp to maintain conversation order
            session_traces.sort(key=lambda t: getattr(t, 'timestamp', datetime.min))

            # Filter traces to valid types and deduplicate by trace_id
            seen_trace_ids = set()
            valid_traces = []
            for trace in session_traces:
                trace_id = getattr(trace, 'id', None)
                if trace_id in seen_trace_ids:
                    continue  # Skip duplicate trace
                seen_trace_ids.add(trace_id)

                trace_metadata = getattr(trace, 'metadata', {}) or {}
                trace_type = trace_metadata.get('mode', '')
                if trace_type in VALID_TYPES:
                    valid_traces.append(trace)

            if not valid_traces:
                logger.debug(f"Session {session_id} has no traces with valid types {VALID_TYPES}")
                return None

            # Extract conversation turns from all valid traces in the session
            all_conversation_turns = []

            for trace_idx, trace in enumerate(valid_traces):
                try:
                    # Extract conversation from this trace
                    trace_conversation = self._extract_conversation_from_spans(getattr(trace, 'spans', []))

                    # If no conversation in spans, try trace input/output
                    if not trace_conversation:
                        trace_conversation = self._extract_conversation_from_trace(trace)

                    if trace_conversation:
                        all_conversation_turns.extend(trace_conversation)

                    # Debug: Log first few extracted turns
                    if trace_idx < 3:
                        for turn in trace_conversation:
                            content_preview = turn.content[:80] if turn.content else "empty"
                            logger.debug(f"  Trace {trace_idx}: {turn.role} = '{content_preview}'...")

                except Exception as e:
                    logger.warning(f"Failed to extract conversation from trace {getattr(trace, 'id', 'unknown')} in session {session_id}: {e}")
                    continue

            if not all_conversation_turns:
                logger.warning(f"Session {session_id} has no conversation data")
                return None

            # Get the LAST trace (most recent) for score association
            last_trace = valid_traces[-1]
            last_trace_id = getattr(last_trace, 'id', None)
            last_trace_timestamp = getattr(last_trace, 'timestamp', None)

            # Get metadata from the first trace
            first_trace = valid_traces[0]
            user_id = getattr(first_trace, 'user_id', None)
            metadata = getattr(first_trace, 'metadata', {}) or {}
            created_at = getattr(first_trace, 'timestamp', datetime.now())

            # Create conversation trace for the session
            return ConversationTrace(
                trace_id=last_trace_id,  # Use LAST trace ID for score association
                user_id=user_id,
                session_id=session_id,
                conversation=all_conversation_turns,
                metadata=metadata,
                created_at=created_at,
                total_turns=len(all_conversation_turns),
                last_trace_timestamp=last_trace_timestamp  # For score timestamp
            )
            
        except Exception as e:
            logger.error(f"Error processing session {session_id}: {e}")
            return None

    def _process_trace(self, trace) -> Optional[ConversationTrace]:
        """Process a single trace into a conversation"""
        try:
            # Safely access trace attributes
            trace_id = getattr(trace, 'id', None)
            if not trace_id:
                logger.warning("Trace has no ID, skipping")
                return None
                
            user_id = getattr(trace, 'user_id', None)
            session_id = getattr(trace, 'session_id', None)
            metadata = getattr(trace, 'metadata', {}) or {}
            created_at = getattr(trace, 'timestamp', None)
            
            if not created_at:
                logger.warning(f"Trace {trace_id} has no timestamp, skipping")
                return None
            
            # Extract conversation from spans first
            spans = getattr(trace, 'spans', []) or []
            conversation = self._extract_conversation_from_spans(spans)
            
            # If no conversation found in spans, try trace input/output directly
            if not conversation:
                logger.info(f"Trace {trace_id} has no spans, checking trace input/output directly")
                conversation = self._extract_conversation_from_trace(trace)
            
            if not conversation:
                logger.debug(f"Trace {trace_id} has no conversation data")
                return None
            
            return ConversationTrace(
                trace_id=trace_id,
                user_id=user_id,
                session_id=session_id,
                conversation=conversation,
                metadata=metadata,
                created_at=created_at,
                total_turns=len(conversation)
            )
            
        except Exception as e:
            logger.error(f"Error processing trace: {e}")
            return None
    
    def _extract_conversation_from_spans(self, spans: List) -> List[ConversationTurn]:
        """Extract conversation turns from Langfuse spans"""
        conversation = []
        
        for span in spans:
            try:
                # Safely access span attributes
                span_name = getattr(span, 'name', '') or ''
                input_data = getattr(span, 'input', {}) or {}
                output_data = getattr(span, 'output', {}) or {}
                span_id = getattr(span, 'id', None)
                
                # Extract user messages (from preprocessing or inference spans)
                if "user" in str(input_data) or "message" in str(input_data):
                    user_content = self._extract_user_message(input_data)
                    if user_content:
                        conversation.append(ConversationTurn(
                            role="user",
                            content=user_content,
                            span_id=span_id
                        ))
                
                # Extract assistant messages (from inference or postprocessing spans)
                if "assistant" in str(output_data) or "response" in str(output_data):
                    assistant_content = self._extract_assistant_message(output_data)
                    if assistant_content:
                        conversation.append(ConversationTurn(
                            role="assistant",
                            content=assistant_content,
                            span_id=span_id
                        ))
                        
            except Exception as e:
                logger.debug(f"Error processing span: {e}")
                continue
        
        return conversation
    
    def _extract_conversation_from_trace(self, trace) -> List[ConversationTurn]:
        """Extract conversation turns directly from trace input/output"""
        conversation = []
        
        try:
            # Get trace input and output
            input_data = getattr(trace, 'input', {}) or {}
            output_data = getattr(trace, 'output', {}) or {}
            
            logger.debug(f"Processing trace with input keys: {list(input_data.keys()) if isinstance(input_data, dict) else type(input_data)}")
            
            # Extract user message from input
            user_content = self._extract_user_message(input_data)
            if user_content:
                conversation.append(ConversationTurn(
                    role="user",
                    content=user_content,
                    span_id=None
                ))
            
            # Extract assistant message from output
            assistant_content = self._extract_assistant_message(output_data)
            if assistant_content:
                conversation.append(ConversationTurn(
                    role="assistant",
                    content=assistant_content,
                    span_id=None
                ))
                
        except Exception as e:
            logger.debug(f"Error extracting conversation from trace: {e}")
        
        return conversation
    
    def _extract_user_message(self, input_data: Any) -> Optional[str]:
        """
        Extract user message from input data.
        """
        if not isinstance(input_data, dict):
            return None

        # Primary path: args[1][0]
        args = input_data.get("args")
        if isinstance(args, list) and len(args) >= 2:
            instances = args[1]

            if isinstance(instances, list) and len(instances) > 0:
                instance = instances[0]

                if isinstance(instance, dict):
                    conv = instance.get("chat", {})
                    if isinstance(conv, dict):
                        # Try current_message first (explicit last user message)
                        current_msg = conv.get("current_message")
                        if isinstance(current_msg, str) and current_msg.strip():
                            return current_msg.strip()

                        # Fall back to last user message in history
                        chat_history = conv.get("history", [])
                        user_info = instance.get("user", {})
                        user_id = user_info.get("user_id") if isinstance(user_info, dict) else None

                        # Find last message from user (by matching user_id or by position)
                        if isinstance(chat_history, list) and len(chat_history) > 0:
                            # Get last message from user
                            for msg in reversed(chat_history):
                                if isinstance(msg, dict):
                                    sender = msg.get("user_id", "")
                                    body = msg.get("body", "")
                                    # Match by user_id
                                    if user_id and sender == user_id:
                                        if isinstance(body, str) and body.strip():
                                            return body.strip()

                            # Fallback: get the last message body
                            last_msg = chat_history[-1]
                            if isinstance(last_msg, dict):
                                body = last_msg.get("body", "")
                                if isinstance(body, str) and body.strip():
                                    return body.strip()

        # Fallback paths
        if isinstance(args, list) and len(args) > 0:
            first_arg = args[0]
            if isinstance(first_arg, list) and len(first_arg) > 0:
                first_arg = first_arg[0]

            if isinstance(first_arg, dict):
                conv = first_arg.get("chat", {})
                if isinstance(conv, dict):
                    current_msg = conv.get("current_message")
                    if isinstance(current_msg, str) and current_msg.strip():
                        return current_msg.strip()

        # Direct field fallbacks
        for field in ["message", "user_message", "input", "prompt", "current_message"]:
            if field in input_data:
                content = input_data[field]
                if isinstance(content, str) and content.strip():
                    return content.strip()

        return None
    
    def _extract_assistant_message(self, output_data: Any) -> Optional[str]:
        """Extract assistant message from output data"""
        if isinstance(output_data, dict):
            # Try different possible field names
            for field in ["response", "output", "content", "text", "result"]:
                if field in output_data:
                    content = output_data[field]
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        
        # Handle list output structure (common in Langfuse traces)
        if isinstance(output_data, list) and len(output_data) > 0:
            first_output = output_data[0]
            if isinstance(first_output, dict):
                # Check for generated_text (chat traces)
                if "generated_text" in first_output and isinstance(first_output["generated_text"], str):
                    return first_output["generated_text"].strip()
                
                # Check for other common fields
                for field in ["response", "output", "content", "text", "result", "raw_result"]:
                    if field in first_output:
                        content = first_output[field]
                        if isinstance(content, str) and content.strip():
                            return content.strip()
        
        return None
    
    def sample_conversation_window(self, conversation: List[ConversationTurn]) -> List[ConversationTurn]:
        """
        Sample conversation window for evaluation.

        Window Selection:
            - turns <= window: Return ALL turns (short session)
            - turns > window: Pick RANDOM mid-point, then take [mid - offset : mid + offset]

        The mid-point is randomly selected (not exact middle) to add diversity.
        We avoid extremes near start/end to ensure room for full window.

        Note: Sessions < min_turns are filtered out in get_traces_for_evaluation()

        Args:
            conversation: Full conversation (already filtered by min_turns)

        Returns:
            Conversation window for evaluation (up to config.conversation_window turns)
        """
        total_turns = len(conversation)

        logger.info(f"Sampling conversation window from {total_turns} total turns")

        # If conversation is shorter than window, return all
        if total_turns <= config.conversation_window:
            logger.info(f"Conversation shorter than window ({total_turns} <= {config.conversation_window}), returning full conversation")
            return conversation

        # Calculate valid range for random mid-point selection
        # Avoid extremes to ensure room for full window on both sides
        min_mid = config.window_offset
        max_mid = total_turns - config.window_offset

        # Pick RANDOM mid-point (not exact middle)
        mid_point = random.randint(min_mid, max_mid)

        # Calculate window boundaries around random mid-point
        start_idx = mid_point - config.window_offset
        end_idx = mid_point + config.window_offset

        # Safety clamp (should not be needed if min_mid/max_mid calculated correctly)
        start_idx = max(0, start_idx)
        end_idx = min(total_turns, end_idx)

        sampled_window = conversation[start_idx:end_idx]
        logger.info(f"Sampled window: random mid={mid_point}, turns {start_idx+1}-{end_idx} ({len(sampled_window)} turns)")

        return sampled_window
    
    def push_evaluation_score(self, trace_id: str, scores: Dict[str, float],
                            conversation_window: List[ConversationTurn],
                            trace_timestamp: Optional[datetime] = None) -> bool:
        """
        Push evaluation scores back to Langfuse using the SDK

        Args:
            trace_id: The trace ID to attach scores to (latest trace in session)
            scores: Dictionary of evaluation scores
            conversation_window: The conversation window that was evaluated
            trace_timestamp: The trace's timestamp for time-series dashboards

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use the Langfuse SDK to create scores
            # NOTE: Environment is set via LANGFUSE_TRACING_ENVIRONMENT at client init, not per-score
            for metric_name, score_value in scores.items():
                score_kwargs = {
                    "trace_id": trace_id,
                    "name": metric_name,
                    "value": score_value,
                    "comment": f"Evaluated {len(conversation_window)} turns | score_env={config.score_environment} | source_env={config.source_environment}"
                }
                # Add timestamp if provided (for time-series dashboards)
                if trace_timestamp:
                    score_kwargs["timestamp"] = trace_timestamp

                self.langfuse.create_score(**score_kwargs)

            logger.info(f"Successfully pushed evaluation for trace {trace_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to push evaluation for trace {trace_id}: {e}")
            return False 
