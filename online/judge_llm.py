import logging
import time
from typing import Dict, Any, List
from replicate import Client
import httpx
import os

from .config import config

logger = logging.getLogger(__name__)

# =============================================================================
# 4 Core Metrics - One prompt per metric, one LLM call per metric
# =============================================================================

METRIC_PROMPTS = {
    "relevance": """You are evaluating a conversation between a user and an AI assistant.

METRIC: RELEVANCE
DEFINITION: Does the AI's response directly address what the user said?

SCALE:
1 - Completely off-topic, ignores user's message
2 - Barely related, misses main point
3 - Partially relevant, addresses some aspects
4 - Mostly relevant, addresses main point well
5 - Perfectly relevant, directly addresses everything user said

CONVERSATION:
{conversation}

TASK: Rate ONLY the relevance of the AI's responses. Consider all turns.
OUTPUT: Respond with ONLY a single number 1-5 (no explanation, no text, just the number).""",

    "engagement": """You are evaluating a conversation between a user and an AI assistant.

METRIC: ENGAGEMENT
DEFINITION: Does the AI's response move the conversation forward and keep the user interested?

SCALE:
1 - Dead-end responses, kills conversation momentum
2 - Minimal effort, barely keeps conversation going
3 - Adequate, maintains conversation but not exciting
4 - Engaging, adds interest and invites continued interaction
5 - Highly engaging, creates compelling dialogue that draws user in

CONVERSATION:
{conversation}

TASK: Rate ONLY how engaging the AI's responses are. Consider all turns.
OUTPUT: Respond with ONLY a single number 1-5 (no explanation, no text, just the number).""",

    "naturalness": """You are evaluating a conversation between a user and an AI assistant.

METRIC: NATURALNESS
DEFINITION: Does the conversation feel natural and human-like, not robotic or scripted?

SCALE:
1 - Robotic, formulaic, obviously AI-generated
2 - Stilted, unnatural phrasing or responses
3 - Acceptable, mostly natural with some awkward moments
4 - Natural, flows like human conversation
5 - Perfectly natural, indistinguishable from human chat

CONVERSATION:
{conversation}

TASK: Rate ONLY how natural the conversation feels. Consider all turns.
OUTPUT: Respond with ONLY a single number 1-5 (no explanation, no text, just the number).""",

    "appropriateness": """You are evaluating a conversation between a user and an AI assistant.

METRIC: APPROPRIATENESS
DEFINITION: Does the AI match the user's tone and intent? (e.g., playful when user is playful, serious when serious, empathetic when user is upset)

SCALE:
1 - Completely mismatched tone, inappropriate responses
2 - Often misreads user's intent or mood
3 - Sometimes matches tone, sometimes misses
4 - Usually matches user's tone and intent well
5 - Perfectly attuned to user's mood and intent throughout

CONVERSATION:
{conversation}

TASK: Rate ONLY how well the AI matches the user's tone/intent. Consider all turns.
OUTPUT: Respond with ONLY a single number 1-5 (no explanation, no text, just the number).""",
}

# Order of evaluation (for logging clarity)
METRIC_ORDER = ["relevance", "engagement", "naturalness", "appropriateness"]


class JudgeLLM:
    """Judge LLM client using Replicate for conversation evaluation.

    Evaluates conversations using core metrics, each with a separate LLM call.
    """

    def __init__(self):
        self.api_token = config.replicate_api_token
        self.model = config.judge_model
        self.timeout = config.replicate_timeout
        self.use_sync = config.replicate_use_sync

        if not self.api_token:
            raise ValueError("Replicate API token not configured")

        os.environ["REPLICATE_API_TOKEN"] = self.api_token

        self.client = Client(
            api_token=self.api_token,
            timeout=httpx.Timeout(self.timeout, connect=30.0)
        )
        mode_str = "sync" if self.use_sync else "stream"
        logger.info(f"JudgeLLM initialized: model={self.model.split(':')[0]}, timeout={self.timeout}s, mode={mode_str}")

    def evaluate_conversation(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate a conversation using metrics (one LLM call per metric).

        Args:
            conversation: List of conversation turns with 'role' and 'content' keys

        Returns:
            Dictionary with metrics dict, composite_score, and timing info
        """
        formatted_conversation = self._format_conversation(conversation)

        # Debug: Show conversation being evaluated
        logger.debug("=== CONVERSATION FOR EVALUATION ===")
        for line in formatted_conversation.split('\n')[:10]:
            safe_line = line.encode('ascii', errors='replace').decode('ascii')
            logger.debug(f"  {safe_line[:150]}..." if len(safe_line) > 150 else f"  {safe_line}")

        logger.info(f"Evaluating {len(conversation)} turns with metrics (sequential)")

        # Evaluate each metric sequentially
        metrics = {}
        total_time = 0.0

        for metric_name in METRIC_ORDER:
            # Let exceptions propagate - caller handles failures with success=False
            score, elapsed = self._evaluate_single_metric(metric_name, formatted_conversation)
            metrics[metric_name] = score
            total_time += elapsed
            logger.info(f"  {metric_name}: {score} ({elapsed:.1f}s)")

        # Calculate composite score (sum of 4 metrics, range 4-20)
        composite_score = sum(metrics.values())

        logger.info(f"Evaluation complete: composite={composite_score}, total_time={total_time:.1f}s")

        return {
            "metrics": metrics,
            "composite_score": composite_score,
            "total_time_seconds": total_time,
        }

    def _evaluate_single_metric(self, metric_name: str, formatted_conversation: str) -> tuple[float, float]:
        """
        Evaluate a single metric using Judge LLM.

        Args:
            metric_name: Name of the metric (relevance, engagement, etc.)
            formatted_conversation: Pre-formatted conversation string

        Returns:
            Tuple of (score, elapsed_time_seconds)
        """
        prompt_template = METRIC_PROMPTS.get(metric_name)
        if not prompt_template:
            raise ValueError(f"Unknown metric: {metric_name}")

        prompt = prompt_template.format(conversation=formatted_conversation)

        start_time = time.time()

        if self.use_sync:
            output = self._call_replicate_sync(prompt)
        else:
            output = self._call_replicate_stream(prompt)

        elapsed = time.time() - start_time

        # Parse single number from response
        score = self._parse_single_score(output)

        return score, elapsed

    def _call_replicate_sync(self, prompt: str) -> str:
        """Call Replicate using sync mode."""
        output = self.client.run(
            self.model,
            input={
                "prompt": prompt,
                "temperature": 0,
                "max_new_tokens": 16,  # Only need a single number
            }
        )
        result = ""
        for chunk in output:
            result += str(chunk)
        return result

    def _call_replicate_stream(self, prompt: str) -> str:
        """Call Replicate using stream mode."""
        output = ""
        for event in self.client.stream(
            self.model,
            input={
                "prompt": prompt,
                "temperature": 0,
                "max_new_tokens": 16,
            }
        ):
            output += str(event)
        return output

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for evaluation prompt."""
        formatted = []
        for i, turn in enumerate(conversation, 1):
            role = turn.get("role", "unknown")
            content = turn.get("content", "").strip()
            if content:
                formatted.append(f"Turn {i} - {role.title()}: {content}")
        return "\n".join(formatted)

    def _parse_single_score(self, response_text: str) -> float:
        """
        Parse a single score (1-5) from LLM response.

        Handles various response formats:
        - "3"
        - "3."
        - "Score: 3"
        - "The score is 3"
        """
        response_text = response_text.strip()

        # Try to find a number 1-5 in the response
        import re
        matches = re.findall(r'\b([1-5])\b', response_text)

        if matches:
            # Take the first valid score found
            score = float(matches[0])
            return max(1.0, min(5.0, score))

        # Fallback: try to parse the whole response as a number
        try:
            score = float(response_text.split()[0].rstrip('.'))
            return max(1.0, min(5.0, score))
        except (ValueError, IndexError):
            raise ValueError(f"Could not parse score from LLM response: '{response_text[:100]}'")
