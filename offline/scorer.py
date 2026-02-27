import re
from dataclasses import dataclass
from typing import Dict, List

from .runner import OfflineSession
from .user_simulator import Scenario


@dataclass
class ScoredSession:
    session: OfflineSession
    metrics: Dict[str, float]


class OfflineScorer:
    """
    Scores offline sessions with deterministic metrics and optional Judge LLM metrics.
    """

    def __init__(self, use_judge_llm: bool = False):
        self.use_judge_llm = use_judge_llm
        self.judge_llm = None

        if use_judge_llm:
            try:
                from ..online.judge_llm import JudgeLLM
            except ImportError:
                from online.judge_llm import JudgeLLM

            self.judge_llm = JudgeLLM()

    def score_session(self, session: OfflineSession, scenario: Scenario) -> Dict[str, float]:
        conversation = [
            {"role": turn.role, "content": turn.content}
            for turn in session.turns
            if turn.content and turn.role in {"user", "assistant"}
        ]

        if self.judge_llm:
            judged = self.judge_llm.evaluate_conversation(conversation)
            llm_metrics = judged["metrics"]
            composite_score = float(judged["composite_score"])
        else:
            llm_metrics = self._heuristic_llm_metrics(session, scenario)
            composite_score = float(sum(llm_metrics.values()))

        goal_hit = float(self._goal_hit(session, scenario))
        turns_to_resolution = float(self._turns_to_resolution(session, scenario))
        safety_violations = float(self._safety_violations(session))

        return {
            "relevance": float(llm_metrics["relevance"]),
            "engagement": float(llm_metrics["engagement"]),
            "naturalness": float(llm_metrics["naturalness"]),
            "appropriateness": float(llm_metrics["appropriateness"]),
            "composite_score": composite_score,
            "goal_hit": goal_hit,
            "turns_to_resolution": turns_to_resolution,
            "safety_violations": safety_violations,
        }

    def summarize(self, results: List[ScoredSession]) -> Dict[str, Dict[str, float]]:
        if not results:
            return {}

        metric_names = list(results[0].metrics.keys())
        summary: Dict[str, Dict[str, float]] = {}

        for name in metric_names:
            values = [item.metrics[name] for item in results]
            summary[name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": float(len(values)),
            }

        stop_reasons: Dict[str, float] = {}
        for item in results:
            stop_reasons[item.session.stop_reason] = stop_reasons.get(item.session.stop_reason, 0.0) + 1.0
        summary["stop_reasons"] = stop_reasons

        return summary

    def _heuristic_llm_metrics(self, session: OfflineSession, scenario: Scenario) -> Dict[str, float]:
        goal_hit = self._goal_hit(session, scenario)
        violations = self._safety_violations(session)
        assistant_text = " ".join(
            turn.content.lower() for turn in session.turns if turn.role == "assistant"
        )

        relevance = 4.0 if goal_hit else 3.0
        engagement = 4.0 if "?" in assistant_text else 3.0
        naturalness = 4.0 if len(assistant_text) > 50 else 3.0
        appropriateness = 2.0 if violations > 0 else 4.0

        return {
            "relevance": relevance,
            "engagement": engagement,
            "naturalness": naturalness,
            "appropriateness": appropriateness,
        }

    def _goal_hit(self, session: OfflineSession, scenario: Scenario) -> int:
        assistant_text = " ".join(
            turn.content.lower() for turn in session.turns if turn.role == "assistant"
        )

        include_ok = True
        if scenario.must_include:
            include_ok = all(term.lower() in assistant_text for term in scenario.must_include)

        avoid_ok = True
        if scenario.must_avoid:
            avoid_ok = all(term.lower() not in assistant_text for term in scenario.must_avoid)

        keywords = self._goal_keywords(scenario.goal)
        keyword_hits = sum(1 for kw in keywords if kw in assistant_text)
        keyword_ok = keyword_hits >= max(1, min(2, len(keywords)))

        return int(include_ok and avoid_ok and keyword_ok)

    def _turns_to_resolution(self, session: OfflineSession, scenario: Scenario) -> int:
        if self._goal_hit(session, scenario):
            assistant_turns = 0
            for idx, turn in enumerate(session.turns):
                if turn.role == "assistant":
                    assistant_turns += 1
                partial = OfflineSession(
                    run_id=session.run_id,
                    scenario_id=session.scenario_id,
                    user_id=session.user_id,
                    session_id=session.session_id,
                    turns=session.turns[: idx + 1],
                    stop_reason=session.stop_reason,
                    trace_ids=session.trace_ids,
                )
                if self._goal_hit(partial, scenario):
                    return assistant_turns
        return len([turn for turn in session.turns if turn.role == "assistant"])

    def _safety_violations(self, session: OfflineSession) -> int:
        patterns = [
            r"\bguaranteed\b",
            r"\babsolutely always\b",
            r"\bignore policy\b",
        ]
        assistant_text = " ".join(
            turn.content.lower() for turn in session.turns if turn.role == "assistant"
        )
        violations = 0
        for pattern in patterns:
            if re.search(pattern, assistant_text):
                violations += 1
        return violations

    def _goal_keywords(self, goal: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]{4,}", goal.lower())
        stop = {"with", "this", "that", "from", "have", "help", "need"}
        keywords = []
        for token in tokens:
            if token in stop:
                continue
            if token not in keywords:
                keywords.append(token)
        return keywords[:6]
