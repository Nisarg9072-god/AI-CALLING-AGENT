"""Calendar tool: schedule_callback."""
from __future__ import annotations
from typing import Any
from app.tools.base import BaseTool, ToolContext, ToolResult
from app.company.service import CompanyService
from app.tools.registry import _make_idempotency_key

_svc = CompanyService()


class ScheduleCallbackTool(BaseTool):
    name = "schedule_callback"
    description = (
        "Schedule a callback for the customer at their preferred time. "
        "Idempotent — calling twice with the same arguments will NOT create duplicates. "
        "Returns callback_id, status, and scheduled time."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
            "preferred_time": {
                "type": "string",
                "description": "Preferred callback time, e.g. 'tomorrow 2pm' or '2026-08-16T14:00'",
            },
            "reason": {
                "type": "string",
                "description": "Brief reason for the callback",
            },
        },
        "required": ["customer_id", "preferred_time", "reason"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        # Generate idempotency key at tool layer as well (belt-and-suspenders).
        # The Registry already checks state-level idempotency; this catches
        # service-level duplicates if the same logical request comes through
        # a different call session.
        idem_key = _make_idempotency_key(self.name, {
            "customer_id": arguments["customer_id"],
            "preferred_time": arguments["preferred_time"],
        })
        callback = _svc.schedule_callback(
            arguments["customer_id"],
            arguments["preferred_time"],
            arguments["reason"],
            idempotency_key=idem_key,
        )
        return ToolResult.ok(callback, tool_name=self.name)
