"""
ToolRegistry — the single gateway for all tool execution.

The Registry is the deterministic control layer between decisions and actions.
It enforces:
  1. Allowlist      — only registered tools may execute
  2. Permissions    — sensitive tools require verified status
  3. Schema         — arguments validated before any execution
  4. Idempotency    — duplicate calls with same key are blocked
  5. Execution      — runs the tool and records the result
  6. Observability  — every execution is recorded in state

The Agent NEVER calls tools directly. All tool calls go through the Registry.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from app.agent.state import CallState, ToolCallRecord
from app.tools.base import BaseTool, ToolContext, ToolResult


class ToolRegistry:
    """
    Registry and execution gateway for all agent tools.

    Usage:
        registry = ToolRegistry()
        registry.register(GetOrderStatusTool())
        result = registry.execute("get_order_status", {"order_id": "ORD-1"}, state)
    """

    def __init__(self, sensitive_tools: list[str] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        # Tools that require customer verification before they can run.
        # Loaded from settings.sensitive_tools — overridable for testing.
        self._sensitive_tools: set[str] = set(sensitive_tools or [])

    # ── Registration ───────────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Raises ValueError on duplicate names."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        if not tool.name:
            raise ValueError(f"Tool {type(tool).__name__} has no name defined.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_available(self, is_verified: bool = False) -> list[str]:
        """
        Return names of tools the agent may currently use.

        Sensitive tools are excluded when the customer is not verified.
        """
        available = []
        for name, tool in self._tools.items():
            if "verified" in tool.required_permissions and not is_verified:
                continue
            if name in self._sensitive_tools and not is_verified:
                continue
            available.append(name)
        return available

    def tool_descriptions_for_prompt(self, is_verified: bool = False) -> list[dict[str, Any]]:
        """Return tool metadata list for injection into the LLM system prompt."""
        available = set(self.list_available(is_verified=is_verified))
        return [
            tool.to_llm_description()
            for name, tool in self._tools.items()
            if name in available
        ]

    # ── Execution pipeline (5 stages) ─────────────────────────────────────────

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        state: CallState,
    ) -> ToolResult:
        """
        Execute a tool through the full validation pipeline.

        Pipeline stages:
          1. Allowlist check     — is the tool registered?
          2. Permission check    — does the caller have required permissions?
          3. Schema validation   — are arguments valid?
          4. Idempotency check   — has this exact call already run?
          5. Execution           — run the tool, record the result

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments (from AgentDecision).
            state: Current call state (read for context, written to record result).

        Returns:
            ToolResult — success or failure, always structured.
        """

        # ── Stage 1: Allowlist ─────────────────────────────────────────────────
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.fail(
                f"Tool '{tool_name}' is not registered. "
                f"Available tools: {list(self._tools.keys())}",
                tool_name=tool_name,
            )

        # ── Stage 2: Permission check ──────────────────────────────────────────
        needs_verification = (
            "verified" in tool.required_permissions
            or tool_name in self._sensitive_tools
        )
        if needs_verification and not state.is_verified:
            return ToolResult.fail(
                f"Tool '{tool_name}' requires customer identity verification. "
                "Please verify the customer first.",
                tool_name=tool_name,
            )

        # ── Stage 3: Schema validation ─────────────────────────────────────────
        valid, error_msg = tool.validate_arguments(arguments)
        if not valid:
            return ToolResult.fail(
                f"Invalid arguments for '{tool_name}': {error_msg}",
                tool_name=tool_name,
            )

        # ── Stage 4: Idempotency check ─────────────────────────────────────────
        idem_key = _make_idempotency_key(tool_name, arguments)
        if state.is_idempotent_duplicate(idem_key):
            # Return the result of the original call (don't re-execute)
            original = next(
                r for r in state.tool_calls_made if r.idempotency_key == idem_key
            )
            return ToolResult.ok(
                {**original.result, "_idempotent": True},
                tool_name=tool_name,
            )

        # ── Stage 5: Execute ───────────────────────────────────────────────────
        context = ToolContext(
            call_id=state.call_id,
            customer_id=state.customer_id,
            is_verified=state.is_verified,
            iteration=state.current_iteration,
        )

        start_ms = time.monotonic() * 1000
        try:
            result = tool.execute(arguments, context)
        except Exception as exc:
            result = ToolResult.fail(
                f"Tool '{tool_name}' raised an unexpected error: {exc}",
                tool_name=tool_name,
            )
        duration_ms = time.monotonic() * 1000 - start_ms
        result.tool_name = tool_name
        result.duration_ms = duration_ms

        # ── Record in state ────────────────────────────────────────────────────
        state.record_tool_call(ToolCallRecord(
            idempotency_key=idem_key,
            tool_name=tool_name,
            arguments=arguments,
            result=result.data if result.success else {"error": result.error},
            success=result.success,
            error=result.error,
            duration_ms=duration_ms,
            iteration=state.current_iteration,
        ))

        return result

    # ── Introspection ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"


# ── Idempotency key generation ─────────────────────────────────────────────────


def _make_idempotency_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Generate a stable, deterministic idempotency key for a tool call.

    SHA-256 of (tool_name + canonical JSON of arguments).
    Same tool + same arguments → same key, always.
    """
    canonical = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
