"""
Calendar tool — schedule_callback.
"""

from __future__ import annotations

from typing import Any

from app.company.service import get_service
from app.tools.base import BaseTool, ToolResult


class ScheduleCallbackTool(BaseTool):
    name = "schedule_callback"
    description = (
        "Schedule a callback for a customer at their preferred time. "
        "Use this when the customer requests to be called back later. "
        "Requires: customer_id, preferred time (any readable format), reason."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "The customer identifier",
            },
            "scheduled_time": {
                "type": "string",
                "description": "Preferred callback time, e.g. '2024-01-20 14:00' or 'tomorrow afternoon'",
            },
            "reason": {
                "type": "string",
                "description": "Brief reason for the callback",
            },
        },
        "required": ["customer_id", "scheduled_time", "reason"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        try:
            result = get_service().schedule_callback(
                customer_id=arguments["customer_id"],
                scheduled_time=arguments["scheduled_time"],
                reason=arguments["reason"],
            )
            return ToolResult.ok(result, tool_name=self.name)
        except Exception as exc:
            return ToolResult.fail(str(exc), tool_name=self.name)
