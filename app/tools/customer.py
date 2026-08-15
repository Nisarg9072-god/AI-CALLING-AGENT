"""Customer tools: get_customer, verify_customer."""
from __future__ import annotations
from typing import Any
from app.tools.base import BaseTool, ToolContext, ToolResult
from app.company.service import CompanyService, CompanyServiceError

_svc = CompanyService()


class GetCustomerTool(BaseTool):
    name = "get_customer"
    description = (
        "Look up a customer's profile by their customer ID. "
        "Returns name, phone, email, and account status. Never returns PIN."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "The customer's ID (e.g. C001)"},
        },
        "required": ["customer_id"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            customer = _svc.get_customer(arguments["customer_id"])
            return ToolResult.ok(customer, tool_name=self.name)
        except CompanyServiceError as e:
            return ToolResult.fail(str(e), tool_name=self.name)


class VerifyCustomerTool(BaseTool):
    name = "verify_customer"
    description = (
        "Verify a customer's identity by checking their PIN. "
        "Must be called before any sensitive operation. "
        "Returns verified=true/false."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
            "pin": {"type": "string", "description": "The customer's 4-digit PIN"},
        },
        "required": ["customer_id", "pin"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            verified = _svc.verify_customer(arguments["customer_id"], arguments["pin"])
            return ToolResult.ok(
                {"verified": verified, "customer_id": arguments["customer_id"]},
                tool_name=self.name,
            )
        except CompanyServiceError as e:
            return ToolResult.fail(str(e), tool_name=self.name)
