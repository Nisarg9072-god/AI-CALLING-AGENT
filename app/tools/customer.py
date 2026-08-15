"""
Customer tools — get_customer, verify_customer.
"""

from __future__ import annotations

from typing import Any

from app.company.service import get_service
from app.tools.base import BaseTool, ToolResult


class GetCustomerTool(BaseTool):
    name = "get_customer"
    description = (
        "Retrieve customer account information by customer ID. "
        "Use this to look up a customer's name, email, phone, and account status."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "The customer's unique identifier (e.g. C001)",
            }
        },
        "required": ["customer_id"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        try:
            service = get_service()
            customer = service.get_customer(arguments["customer_id"])
            return ToolResult.ok(customer, tool_name=self.name)
        except Exception as exc:
            return ToolResult.fail(str(exc), tool_name=self.name)


class VerifyCustomerTool(BaseTool):
    name = "verify_customer"
    description = (
        "Verify a customer's identity using their 4-digit security PIN. "
        "Must be called before any sensitive operations. "
        "Returns verified=true/false."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "The customer's unique identifier",
            },
            "pin": {
                "type": "string",
                "description": "The customer's 4-digit security PIN",
            },
        },
        "required": ["customer_id", "pin"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        try:
            service = get_service()
            verified = service.verify_customer(arguments["customer_id"], arguments["pin"])
            if verified:
                return ToolResult.ok(
                    {"verified": True, "message": "Identity verified successfully."},
                    tool_name=self.name,
                )
            return ToolResult.ok(
                {"verified": False, "message": "Incorrect PIN. Verification failed."},
                tool_name=self.name,
            )
        except Exception as exc:
            return ToolResult.fail(str(exc), tool_name=self.name)
