"""
Policy configuration — all deterministic limits and sensitive tool lists.

Centralized here so they can be tuned without touching agent or harness logic.
"""

from __future__ import annotations

from app.config import settings

# Tools that require verified identity before use
VERIFICATION_REQUIRED_TOOLS: list[str] = settings.sensitive_tools

# All registered tool names (must match BaseTool.name values)
ALL_TOOL_NAMES: list[str] = [
    "get_customer",
    "verify_customer",
    "get_order",
    "get_order_status",
    "get_customer_orders",
    "schedule_callback",
    "create_support_ticket",
    "get_customer_tickets",
    "transfer_to_human",
    "end_call",
]
