"""
Agent — the probabilistic reasoning core.

The Agent's only job: given the current observation, produce a structured
AgentDecision by calling the LLM. It has no direct state mutation.
"""

from __future__ import annotations

from typing import Any

from app.agent.decision import AgentDecision
from app.agent.llm_provider import LLMProvider
from app.agent.prompts import build_observation, build_system_prompt
from app.agent.state import CallState, MessageRole


class Agent:
    """
    Probabilistic reasoning core.

    Reads state → builds prompt → calls LLM → returns structured decision.
    Never writes state. Never calls tools. Never makes deterministic checks.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def decide(
        self,
        state: CallState,
        tool_schemas: list[dict[str, Any]],
        last_tool_result: dict[str, Any] | None = None,
    ) -> AgentDecision:
        """
        Core decision step: Observe → Decide.

        Args:
            state: Current call state (read-only from agent's perspective).
            tool_schemas: Available tools in LLM-friendly format.
            last_tool_result: Result from the previous tool call, if any.

        Returns:
            A structured AgentDecision ready for harness validation.
        """
        system_prompt = build_system_prompt(state, tool_schemas)
        observation = build_observation(state, last_tool_result)

        # Build message list for the LLM
        messages: list[dict[str, str]] = []

        # Add conversation history (up to last 8 messages)
        for msg in state.recent_messages(8):
            role = "user" if msg.role == MessageRole.USER else "assistant"
            messages.append({"role": role, "content": msg.content})

        # Append current observation as a user turn so the LLM sees it
        if last_tool_result is not None or not messages:
            messages.append({"role": "user", "content": observation})

        return self._llm.structured_decision(system_prompt, messages)
