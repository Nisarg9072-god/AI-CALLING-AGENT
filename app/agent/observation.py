"""
Observation — what the agent sees at the start of each iteration.

Design rules:
  - Built by the harness from CallState + last tool result.
  - Contains ONLY what the agent needs to make its next decision.
  - Never contains raw database rows or full CallState.
  - Never contains secrets (API keys, tokens, PINs after verification).
  - Must be serializable to JSON (passed as context to LLM).

This is the 'O' in Observe→Decide→Act.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent.state import CallState, MessageRole


class ToolResultSummary(BaseModel):
    """Summary of the most recent tool execution — fed into the next observation."""
    tool_name: str
    success: bool
    data: dict[str, Any]
    error: str | None = None


class Observation(BaseModel):
    """
    What the agent currently knows — constructed by the harness each iteration.

    This is the complete input to Mistral's reasoning. Nothing more, nothing less.
    The LLM never sees raw state or database contents.
    """

    # ── Current turn ──────────────────────────────────────────────────────────
    iteration: int
    latest_user_message: str

    # ── Recent conversation (last N turns for context) ────────────────────────
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    # Format: [{"role": "user"|"assistant"|"tool", "content": "..."}]

    # ── Tool result from previous action (THE KEY AGENTIC ELEMENT) ────────────
    # When this is populated, the agent's next decision MUST account for it.
    last_tool_result: ToolResultSummary | None = None

    # ── Customer context (populated after get_customer tool runs) ─────────────
    customer_name: str | None = None
    customer_id: str | None = None
    is_verified: bool = False

    # ── Available tools (harness controls this list) ───────────────────────────
    available_tools: list[str] = Field(default_factory=list)

    # ── Limits (so agent can plan) ─────────────────────────────────────────────
    turns_remaining: int = 20
    tool_calls_remaining: int = 10

    # ── Current call metadata ─────────────────────────────────────────────────
    call_id: str = ""
    phone_number: str | None = None


def build_observation(
    state: CallState,
    last_tool_result: ToolResultSummary | None = None,
    recent_message_count: int = 8,
) -> Observation:
    """
    Build an Observation from current CallState.

    Called by the harness at the start of every agent iteration.
    This is the OBSERVE step of the Observe→Decide→Act loop.

    Args:
        state: The current full call state.
        last_tool_result: Result from the previous tool call (if any).
        recent_message_count: How many recent messages to include.

    Returns:
        Observation containing only what the agent needs.
    """
    # Build the recent message list (role + content only, no timestamps)
    recent = [
        {"role": msg.role.value, "content": msg.content}
        for msg in state.recent_messages(recent_message_count)
    ]

    return Observation(
        iteration=state.current_iteration,
        latest_user_message=state.current_user_message,
        recent_messages=recent,
        last_tool_result=last_tool_result,
        customer_name=state.customer_context.get("name"),
        customer_id=state.customer_id,
        is_verified=state.is_verified,
        available_tools=state.available_tools,
        turns_remaining=state.turns_remaining,
        tool_calls_remaining=state.tool_calls_remaining,
        call_id=state.call_id,
        phone_number=state.phone_number,
    )
