"""
CallState — the single source of truth for an entire call session.

Design rules:
  - Only the AgentHarness mutates state. The Agent is read-only.
  - State is fully serializable to JSON (for persistence + tracing).
  - Every field has a clear, documented purpose.
  - No business logic here — pure data.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ── Enumerations ───────────────────────────────────────────────────────────────


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"


class EscalationStatus(str, Enum):
    NONE = "none"
    REQUESTED = "requested"
    ESCALATED = "escalated"


class TerminationReason(str, Enum):
    AGENT_END = "agent_end"
    ESCALATED = "escalated"
    MAX_TURNS_REACHED = "max_turns_reached"
    MAX_TOOL_CALLS_REACHED = "max_tool_calls_reached"
    ERROR = "error"
    USER_HANGUP = "user_hangup"
    TIMEOUT = "timeout"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# ── Sub-models ─────────────────────────────────────────────────────────────────


class ConversationMessage(BaseModel):
    """A single turn in the conversation history."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    """Complete record of a single tool execution."""
    idempotency_key: str
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    success: bool
    error: str | None = None
    duration_ms: float = 0.0
    iteration: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExtractedEntity(BaseModel):
    """An entity extracted from the conversation (order ID, customer ID, etc.)."""
    entity_type: str       # e.g. "order_id", "customer_id", "phone_number"
    value: str
    confidence: float = 1.0
    source_turn: int = 0   # which conversation turn this came from


# ── Main State Model ───────────────────────────────────────────────────────────


class CallState(BaseModel):
    """
    Complete state of a single call session.

    This is passed through every iteration of the agent loop.
    The agent reads it; only the harness writes it.

    Persistence: serialized to SQLite at the end of each turn (Phase 4).
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str | None = None
    phone_number: str | None = None          # E.164 format, e.g. "+919537266092"
    session_started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ── Conversation ──────────────────────────────────────────────────────────
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    latest_user_message: str = ""            # most recent user input
    latest_agent_message: str = ""           # most recent agent response
    current_intent: str | None = None        # inferred by LLM, recorded by harness

    # ── Loop Tracking ─────────────────────────────────────────────────────────
    current_observation: dict[str, Any] | None = None
    current_decision: dict[str, Any] | None = None

    # ── Extracted data ────────────────────────────────────────────────────────
    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)

    # ── Auth / Verification ───────────────────────────────────────────────────
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_attempts: int = 0
    max_verification_attempts: int = 3

    # ── Tool tracking ──────────────────────────────────────────────────────────
    tool_calls_made: list[ToolCallRecord] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)    # set by harness

    # ── Iteration tracking ────────────────────────────────────────────────────
    current_iteration: int = 0
    max_iterations: int = 20                 # overridden from settings.max_turns
    max_tool_calls: int = 10                 # overridden from settings.max_tool_calls

    # ── Escalation ────────────────────────────────────────────────────────────
    escalation_status: EscalationStatus = EscalationStatus.NONE
    escalation_reason: str | None = None

    # ── Termination ───────────────────────────────────────────────────────────
    finished: bool = False
    termination_reason: TerminationReason | None = None
    outcome: str | None = None               # human-readable outcome summary

    # ── Error tracking ────────────────────────────────────────────────────────
    consecutive_errors: int = 0              # resets on success; triggers escalation if too high
    error_log: list[str] = Field(default_factory=list)

    # ── Customer context (populated by get_customer tool) ─────────────────────
    customer_context: dict[str, Any] = Field(default_factory=dict)

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def is_verified(self) -> bool:
        return self.verification_status == VerificationStatus.VERIFIED

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls_made)

    @property
    def turns_remaining(self) -> int:
        return max(0, self.max_iterations - self.current_iteration)

    @property
    def tool_calls_remaining(self) -> int:
        return max(0, self.max_tool_calls - self.tool_call_count)

    @property
    def is_at_limit(self) -> bool:
        return self.current_iteration >= self.max_iterations

    @property
    def last_tool_result(self) -> dict[str, Any] | None:
        if self.tool_calls_made:
            return self.tool_calls_made[-1].result
        return None

    @property
    def last_tool_name(self) -> str | None:
        if self.tool_calls_made:
            return self.tool_calls_made[-1].tool_name
        return None

    # ── Mutation helpers (called only by the harness) ─────────────────────────

    def add_message(self, role: MessageRole, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.conversation_history.append(
            ConversationMessage(role=role, content=content, metadata=metadata or {})
        )

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls_made.append(record)

    def terminate(self, reason: TerminationReason, outcome: str = "") -> None:
        self.finished = True
        self.termination_reason = reason
        self.outcome = outcome

    def is_idempotent_duplicate(self, key: str) -> bool:
        """Check if a tool with this idempotency key has already run."""
        return any(r.idempotency_key == key for r in self.tool_calls_made)

    def get_entity(self, entity_type: str) -> str | None:
        """Return the most recently extracted entity of a given type."""
        matches = [e for e in self.extracted_entities if e.entity_type == entity_type]
        return matches[-1].value if matches else None

    def recent_messages(self, n: int = 10) -> list[ConversationMessage]:
        """Return the last n conversation messages."""
        return self.conversation_history[-n:]
