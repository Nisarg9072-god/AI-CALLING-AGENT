"""
Tool factory — builds and returns a fully-wired ToolRegistry.

Import and call build_tool_registry() anywhere you need a registry instance.
"""

from __future__ import annotations

from app.config import settings
from app.tools.base import BaseTool
from app.tools.calendar import ScheduleCallbackTool
from app.tools.customer import GetCustomerTool, VerifyCustomerTool
from app.tools.escalation import EndCallTool, TransferToHumanTool
from app.tools.orders import GetCustomerOrdersTool, GetOrderStatusTool, GetOrderTool
from app.tools.registry import ToolRegistry
from app.tools.support import CreateSupportTicketTool, GetCustomerTicketsTool


def build_tool_registry(
    sensitive_tools: list[str] | None = None,
) -> ToolRegistry:
    """
    Build a complete ToolRegistry with all tools registered.

    Args:
        sensitive_tools: Override the sensitive tools list (defaults to settings).

    Returns:
        ToolRegistry ready for use by the harness.
    """
    registry = ToolRegistry(
        sensitive_tools=sensitive_tools or settings.sensitive_tools,
    )

    # ── Register all tools ─────────────────────────────────────────────────────
    tools: list[BaseTool] = [
        # Customer
        GetCustomerTool(),
        VerifyCustomerTool(),
        # Orders
        GetOrderTool(),
        GetOrderStatusTool(),
        GetCustomerOrdersTool(),
        # Support
        CreateSupportTicketTool(),
        GetCustomerTicketsTool(),
        # Calendar
        ScheduleCallbackTool(),
        # Escalation
        TransferToHumanTool(),
        EndCallTool(),
    ]

    for tool in tools:
        registry.register(tool)

    return registry


__all__ = ["build_tool_registry"]
