"""
MockLLMProvider — scripted responses for deterministic testing.

Instead of calling an LLM API, returns pre-programmed AgentDecision objects
in sequence. When exhausted, returns a safe END_CALL decision.

Usage:
    provider = MockLLMProvider([
        AgentDecision(action=ActionType.TOOL_CALL, tool_name="get_order_status",
                      arguments={"order_id": "ORD-1001"},
                      reasoning_summary="Customer asked about order."),
        AgentDecision(action=ActionType.SPEAK,
                      response_text="Your order is shipped!",
                      reasoning_summary="Tool returned shipped status."),
        AgentDecision(action=ActionType.END_CALL,
                      response_text="Goodbye!",
                      reasoning_summary="Issue resolved."),
    ])
"""

from __future__ import annotations

from app.agent.decision import ActionType, AgentDecision
from app.agent.observation import Observation
from app.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """
    Scripted LLM provider for tests and eval scenarios.

    Responses are returned in order.
    After exhaustion, returns END_CALL.
    Use .queue() to append more responses at runtime.
    """

    def __init__(self, responses: list[AgentDecision] | None = None) -> None:
        self._responses: list[AgentDecision] = list(responses or [])
        self._index: int = 0
        self.call_count: int = 0        # how many times structured_decision was called
        self.last_observation: Observation | None = None

    def queue(self, decision: AgentDecision) -> None:
        """Append a scripted response."""
        self._responses.append(decision)

    def structured_decision(
        self,
        system_prompt: str,
        observation: Observation,
    ) -> AgentDecision:
        self.call_count += 1
        self.last_observation = observation

        if self._index < len(self._responses):
            decision = self._responses[self._index]
            self._index += 1
            return decision

        # Graceful fallback when all scripted responses are consumed
        return AgentDecision(
            action=ActionType.END_CALL,
            response_text="Thank you for calling. Have a great day!",
            reasoning_summary="MockLLMProvider: all scripted responses consumed.",
            confidence=1.0,
        )

    def reset(self) -> None:
        """Reset to the beginning of the scripted sequence."""
        self._index = 0
        self.call_count = 0
        self.last_observation = None

    @property
    def provider_name(self) -> str:
        return f"MockLLMProvider(queued={len(self._responses)}, used={self._index})"
