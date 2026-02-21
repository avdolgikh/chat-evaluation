import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import config
from .langfuse_client import LangfuseClient, ConversationTrace, ConversationTurn, AbandonmentMetrics
from .judge_llm import JudgeLLM

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of a single conversation evaluation"""
    trace_id: str  # Latest trace ID in session (for score association)
    user_id: Optional[str]
    session_id: Optional[str]
    conversation_window: List[ConversationTurn]
    evaluation_scores: Dict[str, float]
    composite_score: float
    reasoning: Dict[str, str]
    examples: Dict[str, List[str]]
    improvements: Dict[str, List[str]]
    evaluation_timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    trace_timestamp: Optional[datetime] = None  # Trace's timestamp for dashboards

class OnlineEvaluator:
    """Main evaluator class for online evaluation of Langfuse traces"""
    
    def __init__(self):
        """Initialize the evaluator with Langfuse and Judge LLM clients"""
        if not config.validate():
            raise ValueError("Invalid configuration. Please check environment variables.")

        self.langfuse_client = LangfuseClient()
        self.judge_llm = JudgeLLM()
        self._last_abandonment_metrics: Optional[AbandonmentMetrics] = None

        logger.info("Online Evaluator initialized successfully")

    def get_abandonment_metrics(self) -> Optional[AbandonmentMetrics]:
        """Get the abandonment metrics from the last evaluation run"""
        return self._last_abandonment_metrics
    
    def run_daily_evaluation(self, days_back: int = 1,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[EvaluationResult]:
        """
        Run daily evaluation on recent Langfuse traces
        Evaluates conversations grouped by session
        Pushes evaluation scores at the session level

        Args:
            days_back: Number of days back to fetch traces from (used if start/end not provided)
            start_date: Explicit start date for filtering
            end_date: Explicit end date for filtering

        Returns:
            List of evaluation results (one per session)
        """
        logger.info(f"Starting daily evaluation for chat conversations from {days_back} days ago")

        try:
            # Fetch traces from Langfuse (grouped by session)
            conversation_traces = self.langfuse_client.get_traces_for_evaluation(
                days_back=days_back,
                start_date=start_date,
                end_date=end_date
            )

            # Calculate and push session metrics (avg_session_length) - NO Judge LLM needed
            abandonment_metrics = self.langfuse_client.get_abandonment_metrics()
            if abandonment_metrics:
                if config.run_session_metrics:
                    self.langfuse_client.push_abandonment_metrics(abandonment_metrics)
                # Store for summary
                self._last_abandonment_metrics = abandonment_metrics
            else:
                self._last_abandonment_metrics = None

            # If only running session metrics, skip LLM evaluation
            if not config.run_llm_metrics:
                logger.info("Skipping LLM metrics (run_llm_metrics=False)")
                return []

            if not conversation_traces:
                logger.warning("No conversation sessions found for evaluation (all below min_turns threshold)")
                return []

            logger.info(f"Found {len(conversation_traces)} conversation sessions for evaluation")

            # Evaluate each conversation session with Judge LLM
            evaluation_results = []
            successful_evaluations = 0

            for i, conversation_trace in enumerate(conversation_traces, 1):
                logger.info(f"Evaluating conversation session {i}/{len(conversation_traces)}: {conversation_trace.session_id}")
                logger.info(f"  Session {conversation_trace.session_id}: {conversation_trace.total_turns} total turns, user: {conversation_trace.user_id}")

                try:
                    result = self._evaluate_single_conversation_trace(conversation_trace)
                    evaluation_results.append(result)
                    
                    if result.success:
                        successful_evaluations += 1
                        
                        # Log session-level scoring
                        self._log_session_scoring(result)
                        
                        # Push session score back to Langfuse
                        self._push_evaluation_to_langfuse(result)
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate conversation session {conversation_trace.session_id}: {e}")
                    error_result = EvaluationResult(
                        trace_id=conversation_trace.trace_id,
                        user_id=conversation_trace.user_id,
                        session_id=conversation_trace.session_id,
                        conversation_window=[],
                        evaluation_scores={},
                        composite_score=0.0,
                        reasoning={},
                        examples={},
                        improvements={},
                        evaluation_timestamp=datetime.now(),
                        success=False,
                        error_message=str(e),
                        trace_timestamp=conversation_trace.last_trace_timestamp
                    )
                    evaluation_results.append(error_result)
            
            logger.info(f"Evaluation completed: {successful_evaluations}/{len(conversation_traces)} successful conversation sessions")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Failed to run daily evaluation: {e}")
            return []
    
    def _log_session_scoring(self, result: EvaluationResult) -> None:
        """
        Log detailed scoring information for a conversation session
        
        Args:
            result: Evaluation result to log
        """
        try:
            logger.info(f"=== SESSION SCORING LOG ===")
            logger.info(f"Session ID: {result.session_id}")
            logger.info(f"User ID: {result.user_id}")
            logger.info(f"Trace ID: {result.trace_id}")
            logger.info(f"Conversation Window: {len(result.conversation_window)} turns")
            logger.info(f"Session Composite Score: {result.composite_score:.2f}")
            
            # Log individual metric scores
            logger.info("Session Individual Metrics:")
            for metric, score in result.evaluation_scores.items():
                logger.info(f"  {metric}: {score:.2f}")
            
            # Log reasoning if available
            if result.reasoning:
                logger.info("Session Reasoning:")
                for metric, reason in result.reasoning.items():
                    logger.info(f"  {metric}: {reason[:100]}...")  # Truncate long reasoning
            
            # Log examples if available
            if result.examples:
                logger.info("Session Examples:")
                for metric, examples in result.examples.items():
                    if examples:
                        logger.info(f"  {metric}: {examples[0][:100]}...")  # Show first example
            
            # Log improvements if available
            if result.improvements:
                logger.info("Session Improvements:")
                for metric, improvements in result.improvements.items():
                    if improvements:
                        logger.info(f"  {metric}: {improvements[0][:100]}...")  # Show first improvement
            
            logger.info(f"Session Evaluation Timestamp: {result.evaluation_timestamp}")
            logger.info(f"=== END SESSION SCORING ===")
            
        except Exception as e:
            logger.error(f"Failed to log session scoring: {e}")

    def _evaluate_single_conversation_trace(self, conversation_trace: ConversationTrace) -> EvaluationResult:
        """
        Evaluate a single conversation trace (session-level)
        
        Args:
            conversation_trace: Conversation trace from Langfuse (represents a session)
            
        Returns:
            Evaluation result for the session
        """
        try:
            # Sample conversation window
            conversation_window = self.langfuse_client.sample_conversation_window(conversation_trace.conversation)
            
            if not conversation_window:
                raise ValueError("No conversation window could be extracted")
            
            logger.info(f"Evaluating {len(conversation_window)} turns from session {conversation_trace.session_id}")
            
            # Convert to format expected by judge LLM
            conversation_for_evaluation = [
                {"role": turn.role, "content": turn.content}
                for turn in conversation_window
            ]
            
            # Evaluate using judge LLM
            evaluation_result = self.judge_llm.evaluate_conversation(conversation_for_evaluation)
            
            # Create evaluation result
            result = EvaluationResult(
                trace_id=conversation_trace.trace_id,
                user_id=conversation_trace.user_id,
                session_id=conversation_trace.session_id,
                conversation_window=conversation_window,
                evaluation_scores=evaluation_result["metrics"],
                composite_score=evaluation_result["composite_score"],
                reasoning=evaluation_result.get("reasoning", {}),
                examples=evaluation_result.get("examples", {}),
                improvements=evaluation_result.get("improvements", {}),
                evaluation_timestamp=datetime.now(),
                success=True,
                trace_timestamp=conversation_trace.last_trace_timestamp  # For dashboards
            )
            
            logger.info(f"Session evaluation successful for session {conversation_trace.session_id}: composite_score={result.composite_score}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate session {conversation_trace.session_id}: {e}")
            return EvaluationResult(
                trace_id=conversation_trace.trace_id,
                user_id=conversation_trace.user_id,
                session_id=conversation_trace.session_id,
                conversation_window=[],
                evaluation_scores={},
                composite_score=0.0,
                reasoning={},
                examples={},
                improvements={},
                evaluation_timestamp=datetime.now(),
                success=False,
                error_message=str(e),
                trace_timestamp=conversation_trace.last_trace_timestamp
            )
    
    def _push_evaluation_to_langfuse(self, result: EvaluationResult) -> bool:
        """
        Push evaluation result back to Langfuse for the trace

        Args:
            result: Evaluation result to push

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare scores for Langfuse
            scores = {
                "composite_score": result.composite_score,
                **result.evaluation_scores
            }

            # Check dry_run mode
            if config.dry_run:
                logger.info(f"[DRY RUN] Would push scores to Langfuse for trace {result.trace_id}")
                return True

            # Push to Langfuse with trace_id and trace_timestamp
            # trace_id is the latest trace in the session (for proper association)
            # trace_timestamp is used for time-series dashboards (conversation date, not eval run date)
            success = self.langfuse_client.push_evaluation_score(
                trace_id=result.trace_id,
                scores=scores,
                conversation_window=result.conversation_window,
                trace_timestamp=result.trace_timestamp
            )

            if success:
                logger.info(f"Successfully pushed evaluation to Langfuse for trace {result.trace_id}")
            else:
                logger.error(f"Failed to push evaluation to Langfuse for trace {result.trace_id}")

            return success

        except Exception as e:
            logger.error(f"Error pushing evaluation to Langfuse: {e}")
            return False
    
    def get_evaluation_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Generate summary statistics from evaluation results
        
        Args:
            results: List of evaluation results
            
        Returns:
            Summary statistics
        """
        if not results:
            return {"error": "No evaluation results to summarize"}
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                "total_evaluations": len(results),
                "successful_evaluations": 0,
                "success_rate": 0.0,
                "error": "No successful evaluations"
            }
        
        # Calculate summary statistics
        composite_scores = [r.composite_score for r in successful_results]
        metric_scores = {}
        
        for metric in config.llm_metrics:
            scores = [r.evaluation_scores.get(metric, 0.0) for r in successful_results]
            metric_scores[metric] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores)
            }
        
        summary = {
            "total_evaluations": len(results),
            "successful_evaluations": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "composite_score": {
                "mean": sum(composite_scores) / len(composite_scores),
                "min": min(composite_scores),
                "max": max(composite_scores),
                "count": len(composite_scores)
            },
            "metric_scores": metric_scores,
            "evaluation_timestamp": datetime.now().isoformat()
        }

        # Include abandonment metrics if available
        if self._last_abandonment_metrics:
            am = self._last_abandonment_metrics
            summary["abandonment_metrics"] = {
                "total_sessions": am.total_sessions,
                "abandonment_rate": am.abandonment_rate,
                "short_session_rate": am.short_session_rate,
                "engaged_session_rate": am.engaged_session_rate,
                "avg_session_length": am.avg_session_length,
                "median_session_length": am.median_session_length,
                "min_turns": am.min_turns,
                "max_turns": am.max_turns
            }

        return summary
    
    def save_evaluation_results(self, results: List[EvaluationResult], 
                              output_path: str = "evaluation_results.json") -> bool:
        """
        Save evaluation results to a JSON file
        
        Args:
            results: List of evaluation results
            output_path: Path to save the results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                serializable_result = {
                    "trace_id": result.trace_id,
                    "user_id": result.user_id,
                    "session_id": result.session_id,
                    "trace_timestamp": result.trace_timestamp.isoformat() if result.trace_timestamp else None,
                    "conversation_window": [
                        {
                            "role": turn.role,
                            "content": turn.content,
                            "timestamp": turn.timestamp.isoformat() if turn.timestamp else None
                        }
                        for turn in result.conversation_window
                    ],
                    "evaluation_scores": result.evaluation_scores,
                    "composite_score": result.composite_score,
                    "reasoning": result.reasoning,
                    "examples": result.examples,
                    "improvements": result.improvements,
                    "evaluation_timestamp": result.evaluation_timestamp.isoformat(),
                    "success": result.success,
                    "error_message": result.error_message
                }
                serializable_results.append(serializable_result)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved {len(results)} evaluation results to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
            return False 