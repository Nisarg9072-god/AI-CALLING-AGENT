"""
LLMProvider abstract interface.

The Agent depends only on LLMProvider — never on MistralProvider directly.
This allows swapping providers without changing any agent code.

Implementations:
  - MistralProvider  : Real Mistral AI (app/llm/mistral_provider.py)
  - MockLLMProvider  : Scripted responses for tests (app/llm/mock_provider.py)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.agent.decision import AgentDecision
from app.agent.observation import Observation


class LLMProvider(ABC):
    """
    Abstract LLM interface.

    The agent calls structured_decision() once per loop iteration.
    The implementation handles all provider-specific details.
    """

    @abstractmethod
    def structured_decision(
        self,
        system_prompt: str,
        observation: Observation,
    ) -> AgentDecision:
        """
        Ask the LLM to produce a structured AgentDecision.

        Args:
            system_prompt: Full system prompt including tool descriptions.
            observation: Current observation (what the agent knows).

        Returns:
            A validated AgentDecision. NEVER raises — returns a fallback decision on error.
        """

    @property
    def provider_name(self) -> str:
        return type(self).__name__
