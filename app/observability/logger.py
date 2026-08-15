"""
Observability — structured event logger with JSON trace files.

Every agent event is:
  1. Printed to the console (Rich-formatted)
  2. Written to a JSON trace file (for analysis and evals)

Events follow the schema from docs:
  CALL_STARTED | USER_MESSAGE | AGENT_OBSERVATION | AGENT_DECISION |
  DECISION_VALIDATED | TOOL_REQUESTED | TOOL_COMPLETED | TOOL_FAILED |
  GUARDRAIL_BLOCKED | AGENT_RESPONSE | ESCALATION | CALL_ENDED
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.theme import Theme

from app.config import settings

_console = Console(theme=Theme({
    "observe": "cyan",
    "decide": "yellow",
    "act": "green",
    "blocked": "red",
    "tool": "blue",
    "end": "magenta",
}))


class TraceLogger:
    """
    Structured event logger for a single call session.

    Usage:
        logger = TraceLogger(call_id="abc-123")
        logger.log("AGENT_DECISION", {"action": "tool_call", "tool_name": "get_order"})
    """

    def __init__(self, call_id: str, silent: bool = False) -> None:
        self.call_id = call_id
        self.silent = silent
        self._events: list[dict[str, Any]] = []
        self._trace_path: Path | None = None

        if settings.trace_to_file:
            trace_dir = Path(settings.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            short_id = call_id[:8]
            self._trace_path = trace_dir / f"trace_{short_id}.json"

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        """Record and display a single event."""
        event = {
            "event_type": event_type,
            "call_id": self.call_id,
            "iteration": payload.get("iteration", 0),
            "payload": {k: v for k, v in payload.items() if k not in ("call_id", "iteration")},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._events.append(event)

        if not self.silent:
            self._print_event(event_type, payload)

        if self._trace_path:
            self._flush()

    def _print_event(self, event_type: str, payload: dict[str, Any]) -> None:
        iteration = payload.get("iteration", "")
        prefix = f"[dim]#{iteration:>2}[/dim] " if iteration else "    "

        if event_type == "CALL_STARTED":
            _console.print(f"\n[bold observe]>>> CALL STARTED[/bold observe] {payload.get('call_id', '')[:8]}")
        elif event_type == "AGENT_OBSERVATION":
            msg = payload.get("user_message", "")
            _console.print(f"{prefix}[observe]OBSERVE[/observe]  {escape(msg[:80])}")
        elif event_type == "AGENT_DECISION":
            action = payload.get("action", "")
            tool = f" → {payload.get('tool_name', '')}" if payload.get("tool_name") else ""
            reason = payload.get("reasoning_summary", "")
            _console.print(f"{prefix}[decide]DECIDE[/decide]   {action}{tool}")
            _console.print(f"         [dim]{escape(reason[:100])}[/dim]")
        elif event_type == "DECISION_VALIDATED":
            _console.print(f"{prefix}[act]VALIDATE[/act] OK")
        elif event_type == "GUARDRAIL_BLOCKED":
            reason = payload.get("reason", payload.get("block_reason", ""))
            _console.print(f"{prefix}[blocked]BLOCKED[/blocked]  {escape(str(reason)[:100])}")
        elif event_type == "TOOL_REQUESTED":
            tool = payload.get("tool_name", "")
            args = payload.get("arguments", {})
            _console.print(f"{prefix}[tool]TOOL[/tool]     {tool}({escape(str(args)[:60])})")
        elif event_type in ("TOOL_COMPLETED", "TOOL_FAILED"):
            ok = event_type == "TOOL_COMPLETED"
            icon = "[act]OK[/act]" if ok else "[blocked]FAIL[/blocked]"
            data = payload.get("data", payload.get("error", ""))
            _console.print(f"{prefix}[tool]RESULT[/tool]   {icon} {escape(str(data)[:80])}")
        elif event_type == "AGENT_RESPONSE":
            text = payload.get("text", "")
            _console.print(f"{prefix}[bold green]AGENT[/bold green]    {escape(text[:100])}")
        elif event_type == "ESCALATION":
            _console.print(f"{prefix}[end]ESCALATE[/end] {escape(payload.get('text', '')[:80])}")
        elif event_type == "CALL_ENDED":
            reason = payload.get("termination_reason", "")
            turns = payload.get("turns", 0)
            _console.print(
                f"\n[bold end]<<< CALL ENDED[/bold end] "
                f"reason={reason} turns={turns} "
                f"tool_calls={payload.get('tool_calls', 0)}"
            )

    def _flush(self) -> None:
        if self._trace_path:
            with open(self._trace_path, "w", encoding="utf-8") as f:
                json.dump(self._events, f, indent=2, default=str)

    def get_events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def save(self) -> Path | None:
        self._flush()
        return self._trace_path


def make_event_callback(call_id: str, silent: bool = False):
    """Factory: returns a (logger, callback) pair for use with AgentLoop."""
    logger = TraceLogger(call_id=call_id, silent=silent)
    return logger, logger.log
