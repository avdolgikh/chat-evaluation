from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import httpx


class BotAdapter(ABC):
    """Adapter interface for the bot under test."""

    @abstractmethod
    def reply(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class EchoBotAdapter(BotAdapter):
    """Simple local bot used for smoke tests."""

    def reply(self, messages: List[Dict[str, str]]) -> str:
        user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break
        return (
            "I understand your request. "
            f"Here is a quick response based on: {user_message[:180]}"
        )


class HttpBotAdapter(BotAdapter):
    """
    HTTP adapter for pre-prod bot services.

    Request payload:
      {"messages": [{"role": "...", "content": "..."}]}

    Response parsing order:
      reply -> response -> output -> content -> text
      OpenAI-like choices[0].message.content
    """

    def __init__(
        self,
        url: str,
        timeout_seconds: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.headers = headers or {}

    def reply(self, messages: List[Dict[str, str]]) -> str:
        payload = {"messages": messages}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        return self._extract_reply(data)

    def _extract_reply(self, data) -> str:
        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            for key in ["reply", "response", "output", "content", "text"]:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                    content = first.get("text")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

        raise ValueError(f"Could not parse bot response payload: {str(data)[:300]}")


def build_bot_adapter(
    mode: str,
    bot_url: Optional[str] = None,
    timeout_seconds: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
) -> BotAdapter:
    if mode == "echo":
        return EchoBotAdapter()
    if mode == "http":
        if not bot_url:
            raise ValueError("--bot-url is required when --bot-mode=http")
        return HttpBotAdapter(
            url=bot_url,
            timeout_seconds=timeout_seconds,
            headers=headers,
        )
    raise ValueError(f"Unsupported bot mode: {mode}")
