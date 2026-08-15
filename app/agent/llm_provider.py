"""
LLM Provider abstraction — keeps the agent core decoupled from any specific LLM.

Implementations:
- OpenAIProvider: Real GPT-4o with structured JSON output
- MockLLMProvider: Scripted responses for testing without API calls
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from app.agent.decision import AGENT_DECISION_SCHEMA, AgentDecision, ActionType
from app.config import settings


# ── Abstract interface ─────────────────────────────────────────────────────────


class LLMProvider(ABC):
    @abstractmethod
    def structured_decision(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> AgentDecision:
        """
        Generate a structured AgentDecision from the LLM.

        Args:
            system_prompt: The agent's system instructions.
            messages: Conversation history in OpenAI message format.

        Returns:
            A validated AgentDecision.
        """


# ── OpenAI Provider ────────────────────────────────────────────────────────────


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or settings.llm_api_key)
        self._model = model or settings.llm_model

    def structured_decision(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> AgentDecision:
        from openai import OpenAIError

        all_messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=all_messages,  # type: ignore[arg-type]
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_decision",
                        "strict": True,
                        "schema": AGENT_DECISION_SCHEMA,
                    },
                },
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            return AgentDecision(**data)
        except OpenAIError as exc:
            # Graceful fallback — ask for clarification rather than crash
            return AgentDecision(
                action=ActionType.ASK_CLARIFICATION,
                response_text="I'm sorry, I had a technical issue. Could you please repeat that?",
                reasoning_summary=f"LLM API error: {exc}",
                confidence=0.0,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            return AgentDecision(
                action=ActionType.ASK_CLARIFICATION,
                response_text="I didn't quite understand. Could you rephrase that?",
                reasoning_summary=f"JSON parse/validation error: {exc}",
                confidence=0.0,
            )


# ── Mock Provider (for testing / no-API-key mode) ─────────────────────────────


class MockLLMProvider(LLMProvider):
    """
    Scripted LLM provider for tests and CI.

    Accepts a list of pre-programmed AgentDecision responses that are
    returned in order. Useful for deterministic eval scenarios.
    """

    def __init__(self, responses: list[AgentDecision] | None = None) -> None:
        self._responses = list(responses or [])
        self._index = 0

    def queue(self, decision: AgentDecision) -> None:
        self._responses.append(decision)

    def structured_decision(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> AgentDecision:
        if self._index < len(self._responses):
            decision = self._responses[self._index]
            self._index += 1
            return decision
        # Default: end the call gracefully
        return AgentDecision(
            action=ActionType.END_CALL,
            response_text="Thank you for calling. Is there anything else I can help you with?",
            reasoning_summary="Mock provider exhausted all scripted responses.",
            confidence=1.0,
        )


# ── Factory ────────────────────────────────────────────────────────────────────


def build_llm_provider(
    provider_name: str | None = None,
    mock_responses: list[AgentDecision] | None = None,
) -> LLMProvider:
    name = provider_name or settings.llm_provider
    if name == "mock":
        return MockLLMProvider(responses=mock_responses)
    if name == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown LLM provider: '{name}'. Choose 'openai' or 'mock'.")
