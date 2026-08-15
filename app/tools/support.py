"""Support tools: create_support_ticket, get_customer_tickets."""
from __future__ import annotations
from typing import Any
from app.tools.base import BaseTool, ToolContext, ToolResult
from app.company.service import CompanyService

_svc = CompanyService()


class CreateSupportTicketTool(BaseTool):
    name = "create_support_ticket"
    description = (
        "Create a support ticket for the customer's issue. "
        "Use when the issue cannot be resolved immediately. "
        "Returns ticket_id and status."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
            "category": {
                "type": "string",
                "description": "Issue category: billing | shipping | technical | account | other",
            },
            "description": {"type": "string", "description": "Brief description of the issue"},
        },
        "required": ["customer_id", "category", "description"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        ticket = _svc.create_support_ticket(
            arguments["customer_id"],
            arguments["category"],
            arguments["description"],
        )
        return ToolResult.ok(ticket, tool_name=self.name)


class GetCustomerTicketsTool(BaseTool):
    name = "get_customer_tickets"
    description = "Get all support tickets for a customer."
    input_schema = {
        "type": "object",
        "properties": {"customer_id": {"type": "string"}},
        "required": ["customer_id"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        tickets = _svc.get_customer_tickets(arguments["customer_id"])
        return ToolResult.ok({"tickets": tickets, "count": len(tickets)}, tool_name=self.name)
