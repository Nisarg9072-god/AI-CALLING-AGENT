"""
Guardrail engine — deterministic validation of every agent decision.

The guardrails run BEFORE any tool is executed or any message is sent.
If a guardrail blocks the action, the agent loop handles the fallback.

All checks here are deterministic (no LLM involved). This is what separates
the harness from the agent: the agent reasons, the harness enforces.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.decision import ActionType, AgentDecision, ValidatedDecision
from app.agent.state import CallState, TerminationReason


# ── Guardrail result ───────────────────────────────────────────────────────────


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""
    validated: ValidatedDecision | None = None

    @classmethod
    def ok(cls, validated: ValidatedDecision) -> "GuardrailResult":
        return cls(allowed=True, validated=validated)

    @classmethod
    def blocked(cls, reason: str) -> "GuardrailResult":
        return cls(allowed=False, reason=reason)


# ── GuardrailEngine ────────────────────────────────────────────────────────────


class GuardrailEngine:
    """
    Deterministic gate that every AgentDecision must pass before execution.

    Checks (in order):
    1. Max turns limit
    2. Max tool calls limit
    3. Tool name allowlist
    4. Argument presence
    5. Authorization (verification required for sensitive tools)
    6. Idempotency (no repeated side-effecting calls)
    """

    def __init__(
        self,
        available_tools: list[str],
        sensitive_tools: list[str] | None = None,
    ) -> None:
        self._available_tools = set(available_tools)
        self._sensitive_tools = set(sensitive_tools or [])

    def validate(self, decision: AgentDecision, state: CallState) -> GuardrailResult:
        """Run all guardrail checks. Returns GuardrailResult."""
        import hashlib, json

        # 1. Max turns
        if state.current_iteration >= state.max_iterations:
            return GuardrailResult.blocked(
                f"Max turns ({state.max_iterations}) reached. Forcing termination."
            )

        # 2. Max tool calls (only for tool_call actions)
        if (
            decision.action == ActionType.TOOL_CALL
            and state.tool_call_count >= state.max_tool_calls
        ):
            return GuardrailResult.blocked(
                f"Max tool calls ({state.max_tool_calls}) reached. Cannot call more tools."
            )

        # 3. Tool allowlist
        if decision.action == ActionType.TOOL_CALL:
            if decision.tool_name not in self._available_tools:
                return GuardrailResult.blocked(
                    f"Tool '{decision.tool_name}' is not in the allowed list: "
                    f"{sorted(self._available_tools)}"
                )

        # 4. Argument presence
        if decision.action == ActionType.TOOL_CALL and decision.tool_name:
            if decision.arguments is None:
                decision = decision.model_copy(update={"arguments": {}})

        # 5. Authorization
        if (
            decision.action == ActionType.TOOL_CALL
            and decision.tool_name in self._sensitive_tools
            and not state.is_verified
        ):
            return GuardrailResult.blocked(
                f"Tool '{decision.tool_name}' requires verified identity. "
                "Please verify the customer first with verify_customer."
            )

        # 6. Idempotency for sensitive tools
        if (
            decision.action == ActionType.TOOL_CALL
            and decision.tool_name in self._sensitive_tools
        ):
            args = decision.arguments or {}
            payload = json.dumps({"tool": decision.tool_name, "args": args}, sort_keys=True)
            idem_key = hashlib.sha256(payload.encode()).hexdigest()[:16]
            if state.is_idempotent_duplicate(idem_key):
                return GuardrailResult.blocked(
                    f"Duplicate sensitive tool call detected for '{decision.tool_name}'. "
                    "This action was already executed."
                )

        # All checks passed — build validated decision
        args = decision.arguments or {}
        payload = json.dumps({"tool": decision.tool_name or "", "args": args}, sort_keys=True)
        idem_key = hashlib.sha256(payload.encode()).hexdigest()[:16]

        validated = ValidatedDecision(
            decision=decision,
            idempotency_key=idem_key,
            iteration=state.current_iteration,
        )
        return GuardrailResult.ok(validated)
