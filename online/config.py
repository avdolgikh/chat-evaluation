import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=False)

@dataclass
class EvaluatorConfig:
    """Configuration for the online evaluation system"""
    
    # Langfuse Configuration
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Environment settings - separate source (fetch traces) from dest (push scores)
    source_environment: str = os.getenv("SOURCE_ENVIRONMENT", "prod")  # Where to fetch traces from
    score_environment: str = os.getenv("SCORE_ENVIRONMENT", "dev")  # Where to push scores
    
    # Replicate Configuration (Judge LLM)
    replicate_api_token: str = os.getenv("REPLICATE_API_TOKEN", "")
    judge_model: str = os.getenv("JUDGE_MODEL", "mikeei/dolphin-2.9-llama3-70b-gguf")
    replicate_timeout: int = int(os.getenv("REPLICATE_TIMEOUT", "120"))  # Timeout in seconds for Replicate API
    replicate_use_sync: bool = os.getenv("REPLICATE_USE_SYNC", "true").lower() == "true"  # Use run() instead of stream()
    
    # Sampling Configuration
    sample_size: int = 20  # Sessions to evaluate PER DAY (e.g., 7-day range = 20*7=140 sessions)
    max_per_user: int = 5  # Max sessions per user_id (diversity)
    max_per_agent: int = 10  # Max sessions per agent_id (diversity)

    # Conversation Window Configuration
    conversation_window: int = 10  # Turns to evaluate per session (upper bound)
    window_offset: int = 5  # +/- offset around middle for sampling
    min_turns: int = 3  # Minimum turns required in session

    # Trace Fetching Configuration
    trace_name_filter: str = "chat_trace"  # Filter for trace name
    user_id_prefix_filter: str = ""  # Optional user_id prefix filter (e.g., offline-eval-service-)
    max_session_pages: int = 10  # Maximum pages to fetch for sessions

    # Execution Mode
    dry_run: bool = False  # If True, evaluate but don't push scores to Langfuse

    # Evaluation Metrics Configuration
    # LLM metrics (require Judge LLM): relevance, engagement, naturalness, appropriateness
    # Derived metrics (calculated from LLM): composite_score
    # Session metrics (no LLM needed): avg_session_length
    llm_metrics: list = None  # LLM-judged metrics (1-5 scale each)
    run_llm_metrics: bool = True  # Whether to run LLM-judged metrics
    run_session_metrics: bool = True  # Whether to run session metrics (avg_session_length)

    def __post_init__(self):
        if self.llm_metrics is None:
            self.llm_metrics = [
                "relevance", "engagement", "naturalness", "appropriateness"
            ]

    @property
    def all_metrics(self) -> list:
        """All 6 metrics: 4 LLM + composite_score + avg_session_length"""
        return self.llm_metrics + ["composite_score", "avg_session_length"]
    
    def validate(self) -> bool:
        """Validate that required configuration is present"""
        required_vars = [
            self.langfuse_secret_key,
            self.langfuse_public_key,
            self.replicate_api_token
        ]
        return all(var for var in required_vars)

# Global configuration instance
config = EvaluatorConfig() 
