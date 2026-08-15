"""LLM provider factory."""
from __future__ import annotations

from app.agent.decision import AgentDecision
from app.config import settings
from app.llm.base import LLMProvider


def build_llm_provider(
    provider_name: str | None = None,
    mock_responses: list[AgentDecision] | None = None,
) -> LLMProvider:
    """
    Build an LLMProvider instance.

    Args:
        provider_name: Override provider (defaults to settings.llm_provider).
        mock_responses: Pre-programmed responses for MockLLMProvider.

    Returns:
        An LLMProvider ready to use.
    """
    name = provider_name or settings.llm_provider

    if name == "mock":
        from app.llm.mock_provider import MockLLMProvider
        return MockLLMProvider(responses=mock_responses)

    if name == "mistral":
        from app.llm.mistral_provider import MistralProvider
        return MistralProvider()

    raise ValueError(
        f"Unknown LLM provider: '{name}'. Choose 'mistral' or 'mock'."
    )
