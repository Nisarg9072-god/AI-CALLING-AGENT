"""
CallState — the single source of truth for a running call session.

All mutable call data lives here. The agent loop reads it to build its
observation and writes back results after each iteration. The harness
controls state transitions; the agent never writes state directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    FAILED = "failed"


class EscalationStatus(str, Enum):
    NONE = "none"
    REQUESTED = "requested"
    TRANSFERRED = "transferred"


class TerminationReason(str, Enum):
    NONE = "none"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    MAX_TURNS_REACHED = "max_turns_reached"
    MAX_TOOL_CALLS_REACHED = "max_tool_calls_reached"
    CUSTOMER_HANGUP = "customer_hangup"
    AGENT_END = "agent_end"
    ERROR = "error"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ── Sub-models ─────────────────────────────────────────────────────────────────


class ConversationMessage(BaseModel):
    """Single message in the conversation history."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    """Record of a completed tool call, used for idempotency checks."""

    idempotency_key: str
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    success: bool
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    iteration: int = 0


class ExecutionEvent(BaseModel):
    """Lightweight trace event recorded during agent loop execution."""

    event_type: str
    iteration: int
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── CallState ──────────────────────────────────────────────────────────────────


class CallState(BaseModel):
    """
    Immutable-friendly state container for a single call session.

    Design principles:
    - Agent reads this; harness writes this.
    - Never accessed by tools directly — tools only receive what they need.
    - All history is append-only (conversation, tool records, events).
    """

    # Identity
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_phone: str = ""
    customer_id: str | None = None

    # Verification
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_attempts: int = 0

    # Conversation
    conversation_history: list[ConversationMessage] = Field(default_factory=list)

    # Semantic understanding (set by harness after each LLM decision)
    current_intent: str | None = None
    extracted_entities: dict[str, Any] = Field(default_factory=dict)

    # Tool tracking
    tool_calls_made: list[ToolCallRecord] = Field(default_factory=list)
    actions_executed: set[str] = Field(default_factory=set)  # idempotency key set

    # Loop control
    current_iteration: int = 0
    max_iterations: int = 20
    max_tool_calls: int = 10

    # Termination
    escalation_status: EscalationStatus = EscalationStatus.NONE
    termination_reason: TerminationReason = TerminationReason.NONE
    call_outcome: str = ""

    # Observability
    execution_events: list[ExecutionEvent] = Field(default_factory=list)

    # Available tools (set at startup by harness)
    available_tools: list[str] = Field(default_factory=list)

    # Timestamps
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None

    model_config = {"arbitrary_types_allowed": True}

    # ── Derived properties ──────────────────────────────────────────────────────

    @property
    def finished(self) -> bool:
        """True when the agent loop should stop."""
        return self.termination_reason != TerminationReason.NONE

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls_made)

    @property
    def is_verified(self) -> bool:
        return self.verification_status == VerificationStatus.VERIFIED

    # ── Mutation helpers (called by harness only) ───────────────────────────────

    def add_message(self, role: MessageRole, content: str, **metadata: Any) -> None:
        self.conversation_history.append(
            ConversationMessage(role=role, content=content, metadata=metadata)
        )

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls_made.append(record)
        self.actions_executed.add(record.idempotency_key)

    def record_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self.execution_events.append(
            ExecutionEvent(
                event_type=event_type,
                iteration=self.current_iteration,
                payload=payload or {},
            )
        )

    def is_idempotent_duplicate(self, idempotency_key: str) -> bool:
        return idempotency_key in self.actions_executed

    def terminate(self, reason: TerminationReason, outcome: str = "") -> None:
        self.termination_reason = reason
        self.call_outcome = outcome
        self.ended_at = datetime.now(timezone.utc)

    def recent_messages(self, n: int = 10) -> list[ConversationMessage]:
        """Return the last n messages for LLM context window."""
        return self.conversation_history[-n:]
