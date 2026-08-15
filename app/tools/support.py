"""
Support tool — create_support_ticket, get_customer_tickets.
"""

from __future__ import annotations

from typing import Any

from app.company.service import get_service
from app.tools.base import BaseTool, ToolResult


class CreateSupportTicketTool(BaseTool):
    name = "create_support_ticket"
    description = (
        "Create a new support ticket for a customer issue. "
        "Use when the issue cannot be resolved immediately and needs follow-up. "
        "Issue types: billing, technical, delivery, general."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "The customer identifier",
            },
            "issue_type": {
                "type": "string",
                "description": "Category: billing | technical | delivery | general",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the customer's issue",
            },
        },
        "required": ["customer_id", "issue_type", "description"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        try:
            result = get_service().create_support_ticket(
                customer_id=arguments["customer_id"],
                issue_type=arguments["issue_type"],
                description=arguments["description"],
            )
            return ToolResult.ok(result, tool_name=self.name)
        except Exception as exc:
            return ToolResult.fail(str(exc), tool_name=self.name)


class GetCustomerTicketsTool(BaseTool):
    name = "get_customer_tickets"
    description = "Retrieve all support tickets for a specific customer."
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "The customer identifier"},
        },
        "required": ["customer_id"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        try:
            tickets = get_service().get_tickets_for_customer(arguments["customer_id"])
            return ToolResult.ok({"tickets": tickets, "count": len(tickets)}, tool_name=self.name)
        except Exception as exc:
            return ToolResult.fail(str(exc), tool_name=self.name)
