"""
Tool base abstraction.

Every tool in the system MUST extend BaseTool and implement execute().

Design rules:
  - Tools receive only (arguments, context) — they NEVER see CallState or LLM output directly.
  - Tools return ToolResult — always structured, never raw strings.
  - Tool authorization is declared via required_permissions, enforced by the Registry.
  - Tool input schema is declared as JSON Schema, validated by the Registry before execute().
  - Tools must be deterministic — same inputs → same outputs (for idempotency).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Tool context ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolContext:
    """
    Minimal context passed to every tool execution.

    Tools receive ONLY what they need — never full CallState.
    This prevents tools from reading conversation history or bypassing auth.
    """
    call_id: str
    customer_id: str | None = None
    is_verified: bool = False
    iteration: int = 0


# ── Tool result ────────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """
    Structured result from a tool execution.

    Always use ToolResult.ok() or ToolResult.fail() — never construct directly.
    The agent observes this result in the next iteration.
    """
    tool_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0

    @classmethod
    def ok(cls, data: dict[str, Any], tool_name: str = "") -> "ToolResult":
        return cls(tool_name=tool_name, success=True, data=data)

    @classmethod
    def fail(cls, error: str, tool_name: str = "") -> "ToolResult":
        return cls(tool_name=tool_name, success=False, data={}, error=error)

    def to_observation_dict(self) -> dict[str, Any]:
        """Summary suitable for embedding in an Observation."""
        if self.success:
            return {"tool": self.tool_name, "success": True, **self.data}
        return {"tool": self.tool_name, "success": False, "error": self.error}


# ── Base tool ──────────────────────────────────────────────────────────────────


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Subclasses must define:
      - name: str              unique tool identifier
      - description: str       shown to the LLM in the system prompt
      - input_schema: dict     JSON Schema for argument validation
      - required_permissions   list of permissions needed (e.g. ["verified"])
      - execute()              the actual implementation

    The Registry calls validate_arguments() before execute().
    Tools should never do their own auth — that's the Registry's job.
    """

    # ── Class-level declarations (override in subclasses) ──────────────────────
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = {}
    required_permissions: list[str] = []   # e.g. ["verified"] for sensitive ops

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate_arguments(self, arguments: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate tool arguments against input_schema.

        Returns (valid: bool, error_message: str).
        error_message is empty string when valid.

        Uses simple JSON Schema validation (type + required fields only).
        Full JSON Schema validation can be added in Phase 8.
        """
        schema = self.input_schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields
        for field_name in required:
            if field_name not in arguments:
                return False, f"Missing required argument: '{field_name}'"

        # Check types
        for field_name, value in arguments.items():
            if field_name in properties:
                expected_type = properties[field_name].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    return False, (
                        f"Argument '{field_name}' must be type '{expected_type}', "
                        f"got '{type(value).__name__}'"
                    )

        return True, ""

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        py_type = type_map.get(expected)
        if py_type is None:
            return True   # unknown type → allow
        # integers must not be bool (Python: bool is subclass of int)
        if expected == "integer" and isinstance(value, bool):
            return False
        return isinstance(value, py_type)

    # ── Execution (implement in subclass) ──────────────────────────────────────

    @abstractmethod
    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """
        Execute the tool with validated arguments.

        Args:
            arguments: Pre-validated tool arguments (already schema-checked).
            context: Minimal call context (call_id, customer_id, is_verified).

        Returns:
            ToolResult — always, even on failure (use ToolResult.fail()).
        """

    # ── Metadata ───────────────────────────────────────────────────────────────

    def to_llm_description(self) -> dict[str, Any]:
        """Return tool metadata formatted for the LLM system prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "requires_verification": "verified" in self.required_permissions,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
