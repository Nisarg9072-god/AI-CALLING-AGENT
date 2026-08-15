"""Escalation tools: transfer_to_human, end_call."""
from __future__ import annotations
from typing import Any
from app.tools.base import BaseTool, ToolContext, ToolResult


class TransferToHumanTool(BaseTool):
    name = "transfer_to_human"
    description = (
        "Transfer the call to a human agent. "
        "Use when: customer explicitly requests a human, issue is too complex, "
        "multiple tool failures occur, or authorization cannot be completed. "
        "This will end the AI agent's session."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why the call is being transferred",
            },
            "summary": {
                "type": "string",
                "description": "Brief summary of the conversation so far for the human agent",
            },
            "priority": {
                "type": "string",
                "description": "Transfer priority: low | normal | high | urgent",
            },
        },
        "required": ["reason", "summary"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult.ok(
            {
                "transferred": True,
                "reason": arguments["reason"],
                "summary": arguments["summary"],
                "priority": arguments.get("priority", "normal"),
                "call_id": context.call_id,
            },
            tool_name=self.name,
        )


class EndCallTool(BaseTool):
    name = "end_call"
    description = (
        "Gracefully end the call after the customer's issue is resolved. "
        "Use when: the customer confirms they are satisfied, says goodbye, "
        "or no further action is needed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "description": "How the call was resolved: resolved | unresolved | escalated | abandoned",
            },
            "summary": {
                "type": "string",
                "description": "One-sentence summary of what was accomplished",
            },
        },
        "required": ["outcome", "summary"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult.ok(
            {
                "ended": True,
                "outcome": arguments["outcome"],
                "summary": arguments["summary"],
                "call_id": context.call_id,
            },
            tool_name=self.name,
        )
