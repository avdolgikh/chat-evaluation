import random
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Scenario:
    id: str
    persona: str
    goal: str
    constraints: List[str]
    max_turns: int
    must_include: List[str] = field(default_factory=list)
    must_avoid: List[str] = field(default_factory=list)


@dataclass
class ConversationState:
    scenario: Scenario
    user_messages_sent: int = 0
    user_messages: List[str] = field(default_factory=list)
    assistant_messages: List[str] = field(default_factory=list)
    goal_reached: bool = False


class UserSimulator:
    """Lightweight user simulator for multi-turn chat scenarios."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def opening_message(self, scenario: Scenario) -> str:
        templates = [
            "Hi, I need help with this: {goal}.",
            "Can you help me? I want to {goal}.",
            "I need support. Goal: {goal}.",
        ]
        message = self.rng.choice(templates).format(goal=scenario.goal.rstrip("."))
        if scenario.constraints:
            message += f" Context: {scenario.constraints[0]}."
        return message

    def next_message(self, state: ConversationState, bot_reply: str) -> str:
        scenario = state.scenario
        state.assistant_messages.append(bot_reply)
        state.goal_reached = self._goal_reached(state)

        if state.goal_reached:
            return "Thanks, that answers my question."

        lower = bot_reply.lower()

        if ("order" in lower and "id" in lower) or ("reference" in lower and "number" in lower):
            return "I do not have the order ID right now. What can I do instead?"

        if "?" in bot_reply:
            return self._answer_question(scenario, bot_reply)

        followups = [
            f"Can you be specific about {scenario.goal}?",
            "Please give me exact steps.",
            "Can you summarize the next action I should take?",
        ]
        return self.rng.choice(followups)

    def should_stop(self, state: ConversationState) -> bool:
        if state.goal_reached:
            return True
        return state.user_messages_sent >= state.scenario.max_turns

    def _answer_question(self, scenario: Scenario, bot_reply: str) -> str:
        lower = bot_reply.lower()
        if "email" in lower:
            return "Use email only; I am on mobile and want a quick flow."
        if "phone" in lower or "call" in lower:
            return "I prefer not to call. Please share chat steps."
        if "when" in lower or "timeline" in lower:
            return "I need a resolution this week."
        if scenario.constraints:
            return f"Constraint: {scenario.constraints[0]}. Please continue."
        return "Please continue with the best available option."

    def _goal_reached(self, state: ConversationState) -> bool:
        messages_joined = " ".join(state.assistant_messages).lower()
        scenario = state.scenario

        include_ok = True
        if scenario.must_include:
            include_ok = all(term.lower() in messages_joined for term in scenario.must_include)

        avoid_violation = False
        if scenario.must_avoid:
            avoid_violation = any(term.lower() in messages_joined for term in scenario.must_avoid)

        goal_keywords = self._keywords(scenario.goal)
        keyword_hits = sum(1 for kw in goal_keywords if kw in messages_joined)
        keyword_ok = keyword_hits >= max(1, min(2, len(goal_keywords)))

        return include_ok and keyword_ok and not avoid_violation

    def _keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
        stop = {"this", "that", "with", "from", "have", "what", "your", "need", "help"}
        unique = []
        for token in tokens:
            if token in stop:
                continue
            if token not in unique:
                unique.append(token)
        return unique[:6]
