"""
AgentDecision — the structured output of every Mistral reasoning step.

Design rules:
  - Every LLM response MUST be parsed into this model before anything executes.
  - The harness validates this before executing any action.
  - Invalid or malformed decisions are NEVER executed — they trigger recovery.
  - Confidence is informational; the harness does NOT use it to bypass guardrails.

This is the 'D' in Observe→Decide→Act.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ── Action types ───────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    SPEAK = "speak"                     # Generate a spoken/text response to the user
    TOOL_CALL = "tool_call"             # Execute a registered tool
    ASK_CLARIFICATION = "ask_clarification"  # Request more info from the user
    WAIT = "wait"                       # Wait for user input (no response needed)
    ESCALATE = "escalate"               # Request human agent transfer
    END_CALL = "end_call"               # Gracefully terminate the call


# ── Decision model ─────────────────────────────────────────────────────────────


class AgentDecision(BaseModel):
    """
    Structured decision produced by Mistral at each agent iteration.

    Mistral outputs this as JSON. The harness validates it before any action runs.

    Validation rules enforced here (not by the LLM):
      - tool_call actions MUST have tool_name
      - speak/ask_clarification/escalate/end_call MUST have response_text
      - confidence must be 0.0–1.0
    """

    action: ActionType
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    response_text: str | None = None
    reasoning_summary: str = Field(
        ...,
        description="Brief explanation of why this decision was made. Always required.",
        min_length=1,
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in this decision (informational only).",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "AgentDecision":
        """Enforce field requirements based on action type."""
        if self.action == ActionType.TOOL_CALL:
            if not self.tool_name:
                raise ValueError(
                    "tool_name is required when action='tool_call'. "
                    "Specify the exact registered tool name."
                )

        if self.action in (
            ActionType.SPEAK,
            ActionType.ASK_CLARIFICATION,
            ActionType.ESCALATE,
            ActionType.END_CALL,
        ):
            if not self.response_text:
                raise ValueError(
                    f"response_text is required when action='{self.action.value}'. "
                    "Provide the text to say to the customer."
                )
        return self


# ── JSON schema for Mistral prompt injection ───────────────────────────────────

# This schema is embedded in the system prompt so Mistral knows what to output.
# It is also used for output validation.

DECISION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [a.value for a in ActionType],
            "description": "What the agent should do next.",
        },
        "tool_name": {
            "type": ["string", "null"],
            "description": "Required when action='tool_call'. Must be a registered tool name.",
        },
        "arguments": {
            "type": ["object", "null"],
            "description": "Tool arguments as a flat JSON object. Required when action='tool_call'.",
        },
        "response_text": {
            "type": ["string", "null"],
            "description": "Text to say to the customer. Required for speak/ask_clarification/escalate/end_call.",
        },
        "reasoning_summary": {
            "type": "string",
            "description": "One-sentence explanation of why you made this decision. Always required.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Your confidence in this decision (0.0–1.0).",
        },
    },
    "required": ["action", "reasoning_summary"],
}


# ── Validation result ──────────────────────────────────────────────────────────


class ValidationResult(BaseModel):
    """Result of harness validation on an AgentDecision."""
    allowed: bool
    decision: AgentDecision
    block_reason: str | None = None      # Why it was blocked (if allowed=False)
    modified: bool = False               # Whether the harness modified the decision
