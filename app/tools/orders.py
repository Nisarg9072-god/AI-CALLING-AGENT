"""
Order tools — get_order, get_order_status, get_customer_orders.
"""

from __future__ import annotations

from typing import Any

from app.company.service import get_service
from app.tools.base import BaseTool, ToolResult


class GetOrderTool(BaseTool):
    name = "get_order"
    description = (
        "Retrieve full details of a specific order by order ID, "
        "including items, total, delivery address, and tracking number."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order identifier (e.g. ORD-1001)",
            }
        },
        "required": ["order_id"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        try:
            order = get_service().get_order(arguments["order_id"])
            return ToolResult.ok(order, tool_name=self.name)
        except Exception as exc:
            return ToolResult.fail(str(exc), tool_name=self.name)


class GetOrderStatusTool(BaseTool):
    name = "get_order_status"
    description = (
        "Get the current status and estimated delivery date of an order. "
        "More concise than get_order — use this when only status info is needed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order identifier",
            }
        },
        "required": ["order_id"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        try:
            status = get_service().get_order_status(arguments["order_id"])
            return ToolResult.ok(status, tool_name=self.name)
        except Exception as exc:
            return ToolResult.fail(str(exc), tool_name=self.name)


class GetCustomerOrdersTool(BaseTool):
    name = "get_customer_orders"
    description = (
        "Retrieve all orders placed by a specific customer. "
        "Useful when the customer refers to 'my order' without specifying an ID."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "The customer identifier",
            }
        },
        "required": ["customer_id"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        try:
            orders = get_service().get_orders_for_customer(arguments["customer_id"])
            return ToolResult.ok({"orders": orders, "count": len(orders)}, tool_name=self.name)
        except Exception as exc:
            return ToolResult.fail(str(exc), tool_name=self.name)
