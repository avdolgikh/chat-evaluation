import json
import logging
import os
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .bot_adapter import BotAdapter
from .user_simulator import ConversationState, Scenario, UserSimulator

logger = logging.getLogger(__name__)


@dataclass
class SessionTurn:
    role: str
    content: str
    timestamp: str


@dataclass
class OfflineSession:
    run_id: str
    scenario_id: str
    user_id: str
    session_id: str
    turns: List[SessionTurn]
    stop_reason: str
    trace_ids: List[str]


class LangfuseTraceWriter:
    """Writes synthetic chat turns as Langfuse traces."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str,
        environment: str,
        trace_name: str = "chat_trace",
        agent_name: str = "offline-user-ai",
    ):
        from langfuse import Langfuse

        self.langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            timeout=60,
        )
        self.environment = environment
        self.trace_name = trace_name
        self.agent_name = agent_name

    def log_turn_pair(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        timestamp: datetime,
    ) -> Optional[str]:
        metadata = {
            "mode": "chat",
            "agent_name": self.agent_name,
            "offline_eval": True,
        }
        trace_kwargs = {
            "name": self.trace_name,
            "user_id": user_id,
            "session_id": session_id,
            "input": {"message": user_message},
            "output": {"response": assistant_message},
            "metadata": metadata,
            "timestamp": timestamp,
            "environment": self.environment,
        }
        try:
            trace_client = self.langfuse.trace(**trace_kwargs)
            trace_id = getattr(trace_client, "id", None)
            return trace_id
        except TypeError:
            # Some SDK variants may not support "environment" as an argument.
            trace_kwargs.pop("environment", None)
            trace_client = self.langfuse.trace(**trace_kwargs)
            return getattr(trace_client, "id", None)

    def flush(self) -> None:
        try:
            self.langfuse.flush()
        except Exception as exc:
            logger.warning(f"Langfuse flush failed: {exc}")


class OfflineRunner:
    """Runs simulated conversations and optionally writes traces to Langfuse."""

    def __init__(
        self,
        bot_adapter: BotAdapter,
        user_simulator: UserSimulator,
        trace_writer: Optional[LangfuseTraceWriter] = None,
    ):
        self.bot_adapter = bot_adapter
        self.user_simulator = user_simulator
        self.trace_writer = trace_writer

    def run(
        self,
        scenarios: List[Scenario],
        n: int,
        seed: int,
        user_prefix: str,
        output_dir: str,
        max_turns_override: Optional[int] = None,
    ) -> List[OfflineSession]:
        if not scenarios:
            raise ValueError("No scenarios provided")

        os.makedirs(output_dir, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rng = random.Random(seed)

        selected = self._select_scenarios(scenarios, n, rng)
        sessions: List[OfflineSession] = []

        for idx, scenario in enumerate(selected, start=1):
            max_turns = max_turns_override or scenario.max_turns
            session = self._run_single(
                run_id=run_id,
                scenario=scenario,
                max_turns=max_turns,
                user_id=f"{user_prefix}-{run_id}-{idx:04d}",
            )
            sessions.append(session)
            self._save_session(output_dir, session)

        if self.trace_writer:
            self.trace_writer.flush()

        return sessions

    def _run_single(
        self,
        run_id: str,
        scenario: Scenario,
        max_turns: int,
        user_id: str,
    ) -> OfflineSession:
        session_id = f"{scenario.id}-{uuid.uuid4().hex[:10]}"
        state = ConversationState(scenario=scenario)
        turns: List[SessionTurn] = []
        trace_ids: List[str] = []
        stop_reason = "max_turns_exhausted"

        user_message = self.user_simulator.opening_message(scenario)

        while state.user_messages_sent < max_turns:
            now = datetime.now(timezone.utc)
            turns.append(SessionTurn(role="user", content=user_message, timestamp=now.isoformat()))
            state.user_messages.append(user_message)
            state.user_messages_sent += 1

            try:
                bot_reply = self.bot_adapter.reply(
                    [{"role": turn.role, "content": turn.content} for turn in turns]
                )
            except Exception as exc:
                logger.error(f"Bot adapter failed for scenario {scenario.id}: {exc}")
                turns.append(
                    SessionTurn(
                        role="assistant",
                        content=f"[BOT_ERROR] {exc}",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )
                stop_reason = "bot_error"
                break

            assist_ts = datetime.now(timezone.utc)
            turns.append(
                SessionTurn(
                    role="assistant",
                    content=bot_reply,
                    timestamp=assist_ts.isoformat(),
                )
            )

            if self.trace_writer:
                try:
                    trace_id = self.trace_writer.log_turn_pair(
                        user_id=user_id,
                        session_id=session_id,
                        user_message=user_message,
                        assistant_message=bot_reply,
                        timestamp=assist_ts,
                    )
                    if trace_id:
                        trace_ids.append(trace_id)
                except Exception as exc:
                    logger.warning(f"Failed to log trace for session {session_id}: {exc}")

            user_message = self.user_simulator.next_message(state, bot_reply)
            if self.user_simulator.should_stop(state):
                stop_reason = "goal_reached" if state.goal_reached else "max_turns_exhausted"
                break

        return OfflineSession(
            run_id=run_id,
            scenario_id=scenario.id,
            user_id=user_id,
            session_id=session_id,
            turns=turns,
            stop_reason=stop_reason,
            trace_ids=trace_ids,
        )

    def _select_scenarios(
        self,
        scenarios: List[Scenario],
        n: int,
        rng: random.Random,
    ) -> List[Scenario]:
        if n <= len(scenarios):
            return rng.sample(scenarios, n)
        return [rng.choice(scenarios) for _ in range(n)]

    def _save_session(self, output_dir: str, session: OfflineSession) -> None:
        payload = asdict(session)
        path = os.path.join(output_dir, f"{session.run_id}_{session.session_id}.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
