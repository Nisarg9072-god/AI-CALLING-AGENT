"""
ToolRegistry — central registry for all agent tools.

Responsibilities:
- Register tools by name
- Enforce tool allowlist (agent can only call registered tools)
- Validate arguments schema before execution
- Check permissions (verified status for sensitive tools)
- Check idempotency (prevent duplicate side-effecting calls)
- Record all executions in CallState
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, TYPE_CHECKING

from app.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from app.agent.state import CallState


class RegistryError(Exception):
    """Raised when a tool call is rejected by the registry."""


class ToolRegistry:
    def __init__(self, sensitive_tools: list[str] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._sensitive_tools: set[str] = set(sensitive_tools or [])

    # ── Registration ───────────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_available(self, is_verified: bool = False) -> list[str]:
        """Return tool names available given current verification status."""
        return [
            name
            for name, tool in self._tools.items()
            if is_verified or "verified" not in tool.required_permissions
        ]

    def get_schemas_for_llm(self, is_verified: bool = False) -> list[dict[str, Any]]:
        """Return schemas for all currently available tools."""
        return [
            tool.get_schema_for_llm()
            for name, tool in self._tools.items()
            if is_verified or "verified" not in tool.required_permissions
        ]

    # ── Execution ──────────────────────────────────────────────────────────────

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        state: "CallState",
    ) -> ToolResult:
        """
        Execute a tool with full validation pipeline.

        Pipeline:
          1. Existence check (allowlist)
          2. Permission check (verification)
          3. Argument schema validation
          4. Idempotency check
          5. Execute
          6. Record result in state
        """
        from app.agent.state import ToolCallRecord

        # 1. Allowlist
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.fail(
                f"Tool '{tool_name}' is not registered. Available: {list(self._tools.keys())}",
                tool_name=tool_name,
            )

        # 2. Permission
        if "verified" in tool.required_permissions and not state.is_verified:
            return ToolResult.fail(
                f"Tool '{tool_name}' requires identity verification first.",
                tool_name=tool_name,
            )

        # 3. Schema validation
        valid, error = tool.validate_arguments(arguments)
        if not valid:
            return ToolResult.fail(
                f"Invalid arguments for '{tool_name}': {error}",
                tool_name=tool_name,
            )

        # 4. Idempotency
        idem_key = self._make_idempotency_key(tool_name, arguments)
        if tool_name in self._sensitive_tools and state.is_idempotent_duplicate(idem_key):
            # Return the previous result instead of re-executing
            for record in reversed(state.tool_calls_made):
                if record.idempotency_key == idem_key:
                    return ToolResult.ok(record.result, tool_name=tool_name)
            return ToolResult.fail("Duplicate call detected but original result not found.")

        # 5. Execute
        context = {
            "call_id": state.call_id,
            "customer_id": state.customer_id,
            "is_verified": state.is_verified,
            "current_iteration": state.current_iteration,
        }
        try:
            result = tool.execute(arguments, context)
        except Exception as exc:
            result = ToolResult.fail(f"Tool execution error: {exc}", tool_name=tool_name)

        # 6. Record
        record = ToolCallRecord(
            idempotency_key=idem_key,
            tool_name=tool_name,
            arguments=arguments,
            result=result.data if result.success else {"error": result.error},
            success=result.success,
            iteration=state.current_iteration,
        )
        state.record_tool_call(record)
        return result

    @staticmethod
    def _make_idempotency_key(tool_name: str, arguments: dict[str, Any]) -> str:
        payload = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
