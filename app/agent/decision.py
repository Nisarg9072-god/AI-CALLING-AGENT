"""
AgentDecision — the structured output every LLM call must return.

This is the contract between probabilistic reasoning (LLM) and deterministic
execution (harness). The LLM must always produce a valid AgentDecision.
The harness validates it before anything gets executed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ── Action types ───────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    SPEAK = "speak"                   # Produce a response to the customer
    TOOL_CALL = "tool_call"           # Call a company data/action tool
    ASK_CLARIFICATION = "ask_clarification"  # Ask customer for more info
    WAIT = "wait"                     # Pause (e.g., waiting for async op)
    ESCALATE = "escalate"             # Transfer to a human agent
    END_CALL = "end_call"             # Gracefully end the call


# ── AgentDecision ──────────────────────────────────────────────────────────────


class AgentDecision(BaseModel):
    """
    Structured output from the LLM for every agent loop iteration.

    All fields are validated before the harness executes anything. This ensures
    the LLM output is always parseable and safe — never free-form text.
    """

    action: ActionType = Field(
        description="What the agent should do next"
    )
    tool_name: str | None = Field(
        default=None,
        description="Name of the tool to call (only when action=tool_call)"
    )
    arguments: dict[str, Any] | None = Field(
        default=None,
        description="Arguments to pass to the tool (only when action=tool_call)"
    )
    response_text: str | None = Field(
        default=None,
        description="Text to speak to the customer (when action=speak/ask_clarification/escalate/end_call)"
    )
    reasoning_summary: str = Field(
        description="Brief explanation of why this decision was made — required for observability"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this decision (0.0–1.0)"
    )

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> "AgentDecision":
        """Ensure the decision is internally consistent."""
        if self.action == ActionType.TOOL_CALL:
            if not self.tool_name:
                raise ValueError("tool_name is required when action=tool_call")
            if self.arguments is None:
                self.arguments = {}
        if self.action in (
            ActionType.SPEAK,
            ActionType.ASK_CLARIFICATION,
            ActionType.ESCALATE,
            ActionType.END_CALL,
        ):
            if not self.response_text:
                raise ValueError(f"response_text is required when action={self.action}")
        return self


# ── Validated decision (post-guardrail) ────────────────────────────────────────


class ValidatedDecision(BaseModel):
    """A decision that has passed guardrail checks. Used by the harness executor."""

    decision: AgentDecision
    idempotency_key: str
    iteration: int

    @property
    def action(self) -> ActionType:
        return self.decision.action

    @property
    def tool_name(self) -> str | None:
        return self.decision.tool_name

    @property
    def arguments(self) -> dict[str, Any]:
        return self.decision.arguments or {}

    @property
    def response_text(self) -> str | None:
        return self.decision.response_text


# ── JSON schema for structured LLM output ──────────────────────────────────────


AGENT_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [a.value for a in ActionType],
        },
        "tool_name": {"type": ["string", "null"]},
        "arguments": {"type": ["object", "null"]},
        "response_text": {"type": ["string", "null"]},
        "reasoning_summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["action", "reasoning_summary"],
    "additionalProperties": False,
}
