"""
AgentLoop — the real Observe→Decide→Act cycle.

This is THE most important file in the entire project.

A reviewer should be able to open this file and immediately understand
the agentic architecture. The loop is NOT hidden inside a framework.

Architecture:
    AgentRuntime (harness)
        │
        └── AgentLoop.run(state)
                │
                └── while not state.finished:
                        │
                        ├── OBSERVE   build_observation(state, last_result)
                        │
                        ├── DECIDE    agent.decide(observation)
                        │
                        ├── VALIDATE  runtime.validate(decision, state)
                        │
                        ├── ACT       runtime.execute(validated, state)
                        │
                        └── UPDATE    state ← result, iteration += 1

The next decision depends on the result of the previous action.
This is what makes the system genuinely agentic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from app.agent.agent import Agent
from app.agent.decision import ActionType, AgentDecision, ValidationResult
from app.agent.observation import Observation, ToolResultSummary, build_observation
from app.agent.state import (
    CallState,
    EscalationStatus,
    MessageRole,
    TerminationReason,
    VerificationStatus,
)
from app.config import settings
from app.tools.registry import ToolRegistry


# ── Loop result ────────────────────────────────────────────────────────────────


@dataclass
class LoopResult:
    """Summary of what the agent loop produced."""
    call_id: str
    termination_reason: TerminationReason
    outcome: str
    turns: int
    tool_calls: int
    duration_ms: float
    final_state: CallState


# ── Event hook type ────────────────────────────────────────────────────────────

EventCallback = Callable[[str, dict], None]   # (event_type, payload) → None


# ── The Agent Loop ─────────────────────────────────────────────────────────────


class AgentLoop:
    """
    The Observe→Decide→Act loop.

    This is the core of the agentic system. It is intentionally simple
    and transparent — a reviewer can trace every step.

    The loop:
      1. Builds an Observation from current state + last tool result
      2. Asks the Agent (Mistral) to produce a decision
      3. Validates the decision (harness policies)
      4. Executes the validated decision
      5. Updates state
      6. Repeats

    The Agent is stateless — it only sees the Observation.
    All state lives in CallState, mutated only by the loop/harness.
    """

    def __init__(
        self,
        agent: Agent,
        registry: ToolRegistry,
        on_event: EventCallback | None = None,
    ) -> None:
        self._agent = agent
        self._registry = registry
        self._on_event = on_event or (lambda etype, payload: None)

    def run(self, state: CallState) -> LoopResult:
        """
        Run the agent loop until the call is finished.

        Args:
            state: Initial call state (already has current_user_message set).

        Returns:
            LoopResult with summary of the call.
        """
        start_time = time.monotonic()
        last_tool_result: ToolResultSummary | None = None

        self._emit("CALL_STARTED", {"call_id": state.call_id})

        # ═══════════════════════════════════════════════════════════════════════
        # THE AGENT LOOP
        # ═══════════════════════════════════════════════════════════════════════

        while not state.finished:

            # ── Deterministic limit check (harness, not LLM) ─────────────────
            if state.is_at_limit:
                self._handle_max_turns(state)
                break

            if state.tool_call_count >= state.max_tool_calls:
                self._handle_max_tool_calls(state)
                break

            state.current_iteration += 1

            # ────────────────────────────────────────────────────────────────
            # STEP 1: OBSERVE
            # Build what the agent currently knows.
            # Crucially: last_tool_result is injected here so the next
            # decision can depend on the previous action's result.
            # ────────────────────────────────────────────────────────────────

            observation = build_observation(state, last_tool_result=last_tool_result)
            state.available_tools = self._registry.list_available(state.is_verified)

            self._emit("AGENT_OBSERVATION", {
                "call_id": state.call_id,
                "iteration": state.current_iteration,
                "user_message": state.current_user_message,
                "last_tool_result": last_tool_result.model_dump() if last_tool_result else None,
                "is_verified": state.is_verified,
                "available_tools": state.available_tools,
            })

            # ────────────────────────────────────────────────────────────────
            # STEP 2: DECIDE
            # The Agent (Mistral) produces a structured decision.
            # This is the ONLY probabilistic step.
            # ────────────────────────────────────────────────────────────────

            decision = self._agent.decide(observation)

            self._emit("AGENT_DECISION", {
                "call_id": state.call_id,
                "iteration": state.current_iteration,
                "action": decision.action.value,
                "tool_name": decision.tool_name,
                "reasoning_summary": decision.reasoning_summary,
                "confidence": decision.confidence,
            })

            # ────────────────────────────────────────────────────────────────
            # STEP 3: VALIDATE
            # The harness deterministically validates the decision.
            # ────────────────────────────────────────────────────────────────

            validation = self._validate(decision, state)

            if not validation.allowed:
                self._emit("GUARDRAIL_BLOCKED", {
                    "call_id": state.call_id,
                    "iteration": state.current_iteration,
                    "reason": validation.block_reason,
                    "original_action": decision.action.value,
                })
                # Use the modified (safe) decision instead
                decision = validation.decision

            else:
                self._emit("DECISION_VALIDATED", {
                    "call_id": state.call_id,
                    "iteration": state.current_iteration,
                    "action": decision.action.value,
                })

            # ────────────────────────────────────────────────────────────────
            # STEP 4: ACT
            # Execute the validated decision.
            # ────────────────────────────────────────────────────────────────

            last_tool_result = self._execute(decision, state)

            # ── Check if action terminated the loop ───────────────────────
            if state.finished:
                break
            
            # If no tool was executed (e.g. SPEAK or ASK_CLARIFICATION), 
            # the agent has addressed the user. Break the loop to wait for user input.
            if last_tool_result is None:
                break

        # ═══════════════════════════════════════════════════════════════════════

        duration_ms = (time.monotonic() - start_time) * 1000

        self._emit("CALL_ENDED", {
            "call_id": state.call_id,
            "termination_reason": state.termination_reason.value if state.termination_reason else "unknown",
            "outcome": state.outcome,
            "turns": state.current_iteration,
            "tool_calls": state.tool_call_count,
            "duration_ms": duration_ms,
        })

        return LoopResult(
            call_id=state.call_id,
            termination_reason=state.termination_reason or TerminationReason.AGENT_END,
            outcome=state.outcome or "",
            turns=state.current_iteration,
            tool_calls=state.tool_call_count,
            duration_ms=duration_ms,
            final_state=state,
        )

    # ── Validate (deterministic harness logic) ────────────────────────────────

    def _validate(self, decision: AgentDecision, state: CallState) -> ValidationResult:
        """
        Deterministic validation of an AgentDecision.

        Enforces:
          - Tool must be registered
          - Sensitive tools require verification
          - Cannot call tools when at tool call limit
        """
        if decision.action == ActionType.TOOL_CALL:
            tool_name = decision.tool_name or ""

            # Tool must exist
            if tool_name not in self._registry:
                return ValidationResult(
                    allowed=False,
                    decision=_ask_clarification("I couldn't perform that action. Could you rephrase?",
                                                f"Unknown tool: {tool_name}"),
                    block_reason=f"Tool '{tool_name}' not registered.",
                )

            # Sensitive tool requires verification
            if tool_name in settings.sensitive_tools and not state.is_verified:
                return ValidationResult(
                    allowed=False,
                    decision=_ask_clarification(
                        "For security, I need to verify your identity first. "
                        "Could you please provide your 4-digit PIN?",
                        f"Blocked: '{tool_name}' requires verification.",
                    ),
                    block_reason=f"Tool '{tool_name}' requires verified identity.",
                )

        return ValidationResult(allowed=True, decision=decision)

    # ── Execute (action dispatch) ─────────────────────────────────────────────

    def _execute(
        self,
        decision: AgentDecision,
        state: CallState,
    ) -> ToolResultSummary | None:
        """
        Execute a validated AgentDecision.

        Returns a ToolResultSummary if a tool ran (fed into next observation),
        or None for speak/clarification/end/escalate actions.
        """
        action = decision.action

        # ── SPEAK ─────────────────────────────────────────────────────────────
        if action == ActionType.SPEAK:
            text = decision.response_text or ""
            state.current_agent_message = text
            state.add_message(MessageRole.ASSISTANT, text)
            self._emit("AGENT_RESPONSE", {
                "call_id": state.call_id,
                "iteration": state.current_iteration,
                "text": text,
            })
            return None

        # ── ASK CLARIFICATION ─────────────────────────────────────────────────
        if action == ActionType.ASK_CLARIFICATION:
            text = decision.response_text or ""
            state.current_agent_message = text
            state.add_message(MessageRole.ASSISTANT, text)
            self._emit("AGENT_RESPONSE", {
                "call_id": state.call_id,
                "iteration": state.current_iteration,
                "text": text,
                "type": "clarification",
            })
            return None

        # ── TOOL CALL ─────────────────────────────────────────────────────────
        if action == ActionType.TOOL_CALL:
            tool_name = decision.tool_name or ""
            arguments = decision.arguments or {}

            self._emit("TOOL_REQUESTED", {
                "call_id": state.call_id,
                "iteration": state.current_iteration,
                "tool_name": tool_name,
                "arguments": arguments,
            })

            tool_result = self._registry.execute(tool_name, arguments, state)

            # Update verification state if verify_customer ran
            if tool_name == "verify_customer" and tool_result.success:
                if tool_result.data.get("verified"):
                    state.verification_status = VerificationStatus.VERIFIED
                    state.customer_context.update({"verified": True})
                else:
                    state.verification_attempts += 1
                    if state.verification_attempts >= state.max_verification_attempts:
                        state.verification_status = VerificationStatus.FAILED

            # Update customer context if get_customer ran
            if tool_name == "get_customer" and tool_result.success:
                state.customer_context.update(tool_result.data)
                if not state.customer_id:
                    state.customer_id = tool_result.data.get("customer_id")

            # Handle escalation tools
            if tool_name == "transfer_to_human" and tool_result.success:
                state.escalation_status = EscalationStatus.ESCALATED
                state.terminate(TerminationReason.ESCALATED, "Transferred to human agent.")

            if tool_name == "end_call" and tool_result.success:
                state.terminate(
                    TerminationReason.AGENT_END,
                    tool_result.data.get("summary", "Call ended by agent."),
                )

            event_type = "TOOL_COMPLETED" if tool_result.success else "TOOL_FAILED"
            self._emit(event_type, {
                "call_id": state.call_id,
                "iteration": state.current_iteration,
                "tool_name": tool_name,
                "success": tool_result.success,
                "data": tool_result.data if tool_result.success else {},
                "error": tool_result.error,
                "duration_ms": tool_result.duration_ms,
            })

            return ToolResultSummary(
                tool_name=tool_name,
                success=tool_result.success,
                data=tool_result.data if tool_result.success else {},
                error=tool_result.error,
            )

        # ── ESCALATE ──────────────────────────────────────────────────────────
        if action == ActionType.ESCALATE:
            text = decision.response_text or "Let me transfer you to a specialist."
            state.current_agent_message = text
            state.add_message(MessageRole.ASSISTANT, text)
            state.escalation_status = EscalationStatus.ESCALATED
            state.terminate(TerminationReason.ESCALATED, "Agent-initiated escalation.")
            self._emit("ESCALATION", {
                "call_id": state.call_id,
                "iteration": state.current_iteration,
                "text": text,
                "reasoning": decision.reasoning_summary,
            })
            return None

        # ── END CALL ──────────────────────────────────────────────────────────
        if action == ActionType.END_CALL:
            text = decision.response_text or "Thank you for calling. Goodbye!"
            state.current_agent_message = text
            state.add_message(MessageRole.ASSISTANT, text)
            state.terminate(TerminationReason.AGENT_END, "Agent ended call gracefully.")
            self._emit("AGENT_RESPONSE", {
                "call_id": state.call_id,
                "iteration": state.current_iteration,
                "text": text,
                "type": "farewell",
            })
            return None

        return None

    # ── Limit handlers ────────────────────────────────────────────────────────

    def _handle_max_turns(self, state: CallState) -> None:
        msg = "I need to transfer you to a specialist now. Thank you for your patience."
        state.add_message(MessageRole.ASSISTANT, msg)
        state.escalation_status = EscalationStatus.ESCALATED
        state.terminate(TerminationReason.MAX_TURNS_REACHED, "Max turns reached.")
        self._emit("GUARDRAIL_BLOCKED", {
            "call_id": state.call_id,
            "reason": "max_turns_reached",
            "turns": state.current_iteration,
        })

    def _handle_max_tool_calls(self, state: CallState) -> None:
        msg = "I've reached the limit for this session. Let me connect you to a specialist."
        state.add_message(MessageRole.ASSISTANT, msg)
        state.escalation_status = EscalationStatus.ESCALATED
        state.terminate(TerminationReason.MAX_TOOL_CALLS_REACHED, "Max tool calls reached.")
        self._emit("GUARDRAIL_BLOCKED", {
            "call_id": state.call_id,
            "reason": "max_tool_calls_reached",
            "tool_calls": state.tool_call_count,
        })

    # ── Event emission ────────────────────────────────────────────────────────

    def _emit(self, event_type: str, payload: dict) -> None:
        try:
            self._on_event(event_type, payload)
        except Exception:
            pass  # Never let observability break the loop


# ── Helpers ────────────────────────────────────────────────────────────────────


def _ask_clarification(text: str, reason: str) -> AgentDecision:
    return AgentDecision(
        action=ActionType.ASK_CLARIFICATION,
        response_text=text,
        reasoning_summary=reason,
        confidence=1.0,
    )
