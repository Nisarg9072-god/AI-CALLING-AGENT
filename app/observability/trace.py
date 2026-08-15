"""
Observability — event types, ExecutionEvent model, and TraceLogger.

Every meaningful step in the agent loop is recorded as an event.
The trace can be displayed in the terminal (rich) and saved to JSON.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ── Event types ────────────────────────────────────────────────────────────────


class EventType(str, Enum):
    CALL_STARTED = "CALL_STARTED"
    USER_MESSAGE = "USER_MESSAGE"
    AGENT_DECISION = "AGENT_DECISION"
    TOOL_REQUESTED = "TOOL_REQUESTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    TOOL_FAILED = "TOOL_FAILED"
    AGENT_RESPONSE = "AGENT_RESPONSE"
    GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"
    VERIFICATION_SUCCESS = "VERIFICATION_SUCCESS"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    ESCALATION = "ESCALATION"
    CALL_ENDED = "CALL_ENDED"
    ERROR = "ERROR"


# ── ExecutionEvent ─────────────────────────────────────────────────────────────


class ExecutionEvent(BaseModel):
    event_type: EventType
    call_id: str
    iteration: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "call_id": self.call_id,
            "iteration": self.iteration,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }


# ── TraceLogger ────────────────────────────────────────────────────────────────


class TraceLogger:
    """
    Records all events during a call session.

    Outputs:
    - Real-time rich console display (shown in CLI)
    - JSON trace file saved to disk (for post-run analysis)
    """

    def __init__(self, call_id: str, trace_dir: str = "./traces") -> None:
        self.call_id = call_id
        self.events: list[ExecutionEvent] = []
        self._trace_dir = Path(trace_dir)
        self._console_enabled = True

    def record(
        self,
        event_type: EventType,
        iteration: int = 0,
        **payload: Any,
    ) -> ExecutionEvent:
        event = ExecutionEvent(
            event_type=event_type,
            call_id=self.call_id,
            iteration=iteration,
            payload=dict(payload),
        )
        self.events.append(event)
        return event

    def save_to_file(self) -> Path | None:
        """Save all events to a JSON file. Returns the file path."""
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            filename = self._trace_dir / f"trace_{self.call_id[:8]}.json"
            data = [e.to_dict() for e in self.events]
            filename.write_text(json.dumps(data, indent=2))
            return filename
        except Exception:
            return None

    def summary(self) -> dict[str, Any]:
        """Return a summary of this trace."""
        event_counts: dict[str, int] = {}
        for e in self.events:
            event_counts[e.event_type.value] = event_counts.get(e.event_type.value, 0) + 1
        return {
            "call_id": self.call_id,
            "total_events": len(self.events),
            "event_counts": event_counts,
            "iterations": max((e.iteration for e in self.events), default=0),
        }
