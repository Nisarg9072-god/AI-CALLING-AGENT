"""
Tool base abstractions — BaseTool and ToolResult.

Tools are the only gateway between the agent and external data/actions.
The agent never touches databases or APIs directly — always through tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ── ToolResult ─────────────────────────────────────────────────────────────────


class ToolResult(BaseModel):
    """Standardized result returned by every tool execution."""

    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    tool_name: str = ""
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def ok(cls, data: dict[str, Any], tool_name: str = "") -> "ToolResult":
        return cls(success=True, data=data, tool_name=tool_name)

    @classmethod
    def fail(cls, error: str, tool_name: str = "") -> "ToolResult":
        return cls(success=False, error=error, tool_name=tool_name)

    def to_context_str(self) -> str:
        """Human-readable summary for inclusion in the LLM context."""
        if self.success:
            return f"[{self.tool_name}] SUCCESS: {self.data}"
        return f"[{self.tool_name}] ERROR: {self.error}"


# ── BaseTool ───────────────────────────────────────────────────────────────────


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Design rules:
    - Tools never access CallState directly (receives only what they need).
    - All I/O is synchronous for simplicity; wrap with asyncio if needed.
    - Required permissions enforce authorization at the registry level.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = {}
    required_permissions: list[str] = []  # e.g. ["verified"] for sensitive ops

    @abstractmethod
    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> ToolResult:
        """
        Execute the tool with validated arguments.

        Args:
            arguments: Tool-specific parameters (pre-validated by registry).
            context: Minimal call context (call_id, customer_id, is_verified, etc.).

        Returns:
            ToolResult with success/failure and structured data.
        """

    def validate_arguments(self, arguments: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate arguments against input_schema.
        Returns (valid, error_message).
        """
        required_fields = self.input_schema.get("required", [])
        properties = self.input_schema.get("properties", {})

        for field in required_fields:
            if field not in arguments:
                return False, f"Missing required field: '{field}'"

        for field, value in arguments.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    return False, f"Field '{field}' must be of type {expected_type}"

        return True, ""

    @staticmethod
    def _check_type(value: Any, expected: str | list[str]) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        if isinstance(expected, list):
            return any(BaseTool._check_type(value, t) for t in expected)
        py_type = type_map.get(expected)
        if py_type is None:
            return True  # Unknown type — pass through
        return isinstance(value, py_type)

    def get_schema_for_llm(self) -> dict[str, Any]:
        """Return the tool schema in a format suitable for the LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "required_permissions": self.required_permissions,
        }
