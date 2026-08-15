"""Order tools: get_order, get_order_status, get_customer_orders."""
from __future__ import annotations
from typing import Any
from app.tools.base import BaseTool, ToolContext, ToolResult
from app.company.service import CompanyService, CompanyServiceError

_svc = CompanyService()


class GetOrderTool(BaseTool):
    name = "get_order"
    description = "Get complete details of an order by order ID."
    input_schema = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "Order ID (e.g. ORD-1001)"},
        },
        "required": ["order_id"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            return ToolResult.ok(_svc.get_order(arguments["order_id"]), tool_name=self.name)
        except CompanyServiceError as e:
            return ToolResult.fail(str(e), tool_name=self.name)


class GetOrderStatusTool(BaseTool):
    name = "get_order_status"
    description = (
        "Get the current status and estimated delivery of an order. "
        "Returns: status, estimated_delivery, tracking_number. "
        "Use this when the customer asks 'where is my order' or 'what is my order status'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "Order ID (e.g. ORD-1001)"},
        },
        "required": ["order_id"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            return ToolResult.ok(_svc.get_order_status(arguments["order_id"]), tool_name=self.name)
        except CompanyServiceError as e:
            return ToolResult.fail(str(e), tool_name=self.name)


class GetCustomerOrdersTool(BaseTool):
    name = "get_customer_orders"
    description = "Get all orders for a customer."
    input_schema = {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
        },
        "required": ["customer_id"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        orders = _svc.get_orders_for_customer(arguments["customer_id"])
        return ToolResult.ok({"orders": orders, "count": len(orders)}, tool_name=self.name)
