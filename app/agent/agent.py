"""
Agent — the probabilistic reasoning component.

This is the ONLY component that calls the LLM.

Design rules:
  - The Agent is READ-ONLY with respect to state.
  - The Agent NEVER executes tools.
  - The Agent NEVER writes to state.
  - The Agent NEVER accesses the database.
  - The Agent's only job: Observation → AgentDecision.

The harness (AgentRuntime) handles all state mutations, tool execution,
validation, and policy enforcement.
"""

from __future__ import annotations

from app.agent.decision import AgentDecision
from app.agent.observation import Observation
from app.agent.prompts import build_system_prompt
from app.llm.base import LLMProvider
from app.tools.registry import ToolRegistry


class Agent:
    """
    The probabilistic reasoning component of the agent loop.

    The Agent wraps an LLMProvider and a ToolRegistry.
    It builds the system prompt and calls the LLM to produce decisions.

    It does NOT:
      - Execute tools
      - Modify state
      - Enforce guardrails
      - Access databases

    All of those are the Harness's responsibilities.
    """

    def __init__(self, llm: LLMProvider, registry: ToolRegistry) -> None:
        self._llm = llm
        self._registry = registry

    def decide(self, observation: Observation) -> AgentDecision:
        """
        The DECIDE step of the Observe→Decide→Act loop.

        Given the current observation (what the agent knows), produce a
        structured decision for the harness to validate and execute.

        Args:
            observation: Current snapshot of the call state.

        Returns:
            AgentDecision — always. Never raises.
        """
        system_prompt = build_system_prompt(self._registry, observation)
        return self._llm.structured_decision(system_prompt, observation.messages)

    @property
    def provider_name(self) -> str:
        return self._llm.provider_name
