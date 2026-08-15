"""
Escalation tools — transfer_to_human, end_call.

These are terminal actions that cause the agent loop to stop.
The harness checks for these and sets the appropriate termination state.
"""

from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool, ToolResult


class TransferToHumanTool(BaseTool):
    name = "transfer_to_human"
    description = (
        "Escalate the call to a human agent. "
        "Use when: the customer explicitly requests a human, the issue is too complex, "
        "the customer is frustrated, or you cannot resolve the problem with available tools. "
        "Always explain why you are transferring."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Reason for escalation — shown to the human agent",
            },
            "summary": {
                "type": "string",
                "description": "Brief summary of the conversation so far",
            },
            "priority": {
                "type": "string",
                "description": "Priority level: low | normal | high | urgent",
            },
        },
        "required": ["reason", "summary"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        return ToolResult.ok(
            {
                "escalated": True,
                "reason": arguments.get("reason", ""),
                "summary": arguments.get("summary", ""),
                "priority": arguments.get("priority", "normal"),
                "message": "Transferring to a human agent now. Please hold.",
            },
            tool_name=self.name,
        )


class EndCallTool(BaseTool):
    name = "end_call"
    description = (
        "Gracefully end the call after the customer's issue is resolved. "
        "Always use this — never just stop responding. "
        "Provide a brief closing message."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "description": "How the call ended: resolved | unresolved | customer_satisfied",
            },
            "summary": {
                "type": "string",
                "description": "One-sentence summary of what was accomplished",
            },
        },
        "required": ["outcome", "summary"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        return ToolResult.ok(
            {
                "ended": True,
                "outcome": arguments.get("outcome", "resolved"),
                "summary": arguments.get("summary", ""),
                "message": "Call ended.",
            },
            tool_name=self.name,
        )
