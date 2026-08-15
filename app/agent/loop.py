"""
AgentLoop — the actual Observe → Decide → Act cycle.

This is the heart of the agentic architecture. Each iteration:
  1. OBSERVE  — build observation from state + last tool result
  2. DECIDE   — LLM produces structured AgentDecision
  3. VALIDATE — guardrails check the decision deterministically
  4. ACT      — execute the decision (speak / call tool / escalate / end)
  5. RECORD   — log everything to trace

The loop continues until state.finished is True.
"""

from __future__ import annotations

from typing import Any

from app.agent.agent import Agent
from app.agent.decision import ActionType, AgentDecision
from app.agent.state import (
    CallState,
    EscalationStatus,
    MessageRole,
    TerminationReason,
    VerificationStatus,
)
from app.harness.guardrails import GuardrailEngine
from app.observability.trace import EventType, TraceLogger
from app.tools.registry import ToolRegistry


class AgentLoop:
    """
    Runs the core Observe→Decide→Act loop for a single call session.

    The AgentRuntime owns this loop and feeds it state + tools.
    """

    def __init__(
        self,
        agent: Agent,
        tool_registry: ToolRegistry,
        guardrail_engine: GuardrailEngine,
        trace_logger: TraceLogger,
    ) -> None:
        self._agent = agent
        self._registry = tool_registry
        self._guardrails = guardrail_engine
        self._trace = trace_logger

    def run(self, state: CallState, initial_message: str) -> CallState:
        """
        Execute the full agent loop for a call.

        Args:
            state: Initial call state (mutated in-place by the loop).
            initial_message: The customer's opening message.

        Returns:
            Final call state after loop terminates.
        """
        # Record the initial customer message
        state.add_message(MessageRole.USER, initial_message)
        self._trace.record(EventType.USER_MESSAGE, iteration=0, content=initial_message)

        last_tool_result: dict[str, Any] | None = None

        while not state.finished:
            state.current_iteration += 1
            iteration = state.current_iteration

            # ── OBSERVE ──────────────────────────────────────────────────────
            tool_schemas = self._registry.get_schemas_for_llm(state.is_verified)

            # ── DECIDE ───────────────────────────────────────────────────────
            decision = self._agent.decide(state, tool_schemas, last_tool_result)
            last_tool_result = None  # consumed

            self._trace.record(
                EventType.AGENT_DECISION,
                iteration=iteration,
                action=decision.action.value,
                tool_name=decision.tool_name,
                reasoning=decision.reasoning_summary,
                confidence=decision.confidence,
            )

            # ── VALIDATE (guardrails) ─────────────────────────────────────────
            guard_result = self._guardrails.validate(decision, state)

            if not guard_result.allowed:
                self._trace.record(
                    EventType.GUARDRAIL_BLOCKED,
                    iteration=iteration,
                    reason=guard_result.reason,
                )
                # If max turns exceeded, force termination
                if state.current_iteration >= state.max_iterations:
                    state.add_message(
                        MessageRole.ASSISTANT,
                        "I apologize, but I need to transfer you to a human agent now.",
                    )
                    state.terminate(TerminationReason.MAX_TURNS_REACHED)
                    break
                # Otherwise, tell the agent about the block and continue
                state.add_message(
                    MessageRole.ASSISTANT,
                    f"[System: Action blocked — {guard_result.reason}]",
                )
                continue

            validated = guard_result.validated
            assert validated is not None

            # ── ACT ───────────────────────────────────────────────────────────
            action = validated.action

            if action == ActionType.SPEAK:
                text = validated.response_text or ""
                state.add_message(MessageRole.ASSISTANT, text)
                self._trace.record(EventType.AGENT_RESPONSE, iteration=iteration, text=text)

            elif action == ActionType.ASK_CLARIFICATION:
                text = validated.response_text or ""
                state.add_message(MessageRole.ASSISTANT, text)
                self._trace.record(EventType.AGENT_RESPONSE, iteration=iteration, text=text)

            elif action == ActionType.TOOL_CALL:
                tool_name = validated.tool_name or ""
                args = validated.arguments

                self._trace.record(
                    EventType.TOOL_REQUESTED,
                    iteration=iteration,
                    tool_name=tool_name,
                    arguments=args,
                )

                result = self._registry.execute(tool_name, args, state)
                last_tool_result = result.model_dump()

                if result.success:
                    self._trace.record(
                        EventType.TOOL_COMPLETED,
                        iteration=iteration,
                        tool_name=tool_name,
                        data=result.data,
                    )
                    # Special handling for verify_customer result
                    if tool_name == "verify_customer":
                        if result.data.get("verified"):
                            state.verification_status = VerificationStatus.VERIFIED
                            self._trace.record(
                                EventType.VERIFICATION_SUCCESS, iteration=iteration
                            )
                        else:
                            state.verification_attempts += 1
                            self._trace.record(
                                EventType.VERIFICATION_FAILED,
                                iteration=iteration,
                                attempt=state.verification_attempts,
                            )

                    # Special handling for terminal tools
                    if tool_name == "transfer_to_human":
                        state.escalation_status = EscalationStatus.TRANSFERRED
                        state.terminate(
                            TerminationReason.ESCALATED,
                            outcome=result.data.get("reason", "Escalated to human"),
                        )
                        self._trace.record(
                            EventType.ESCALATION,
                            iteration=iteration,
                            reason=result.data.get("reason", ""),
                        )
                        break

                    if tool_name == "end_call":
                        state.terminate(
                            TerminationReason.AGENT_END,
                            outcome=result.data.get("summary", "Call ended"),
                        )
                        self._trace.record(
                            EventType.CALL_ENDED,
                            iteration=iteration,
                            outcome=result.data.get("outcome", "resolved"),
                        )
                        break

                else:
                    self._trace.record(
                        EventType.TOOL_FAILED,
                        iteration=iteration,
                        tool_name=tool_name,
                        error=result.error,
                    )

            elif action == ActionType.ESCALATE:
                text = validated.response_text or "Transferring you to a human agent now."
                state.add_message(MessageRole.ASSISTANT, text)
                state.escalation_status = EscalationStatus.REQUESTED
                state.terminate(TerminationReason.ESCALATED, outcome="Escalated by agent")
                self._trace.record(EventType.ESCALATION, iteration=iteration, text=text)
                break

            elif action == ActionType.END_CALL:
                text = validated.response_text or "Thank you for calling. Goodbye!"
                state.add_message(MessageRole.ASSISTANT, text)
                state.terminate(TerminationReason.AGENT_END, outcome="Call ended gracefully")
                self._trace.record(EventType.CALL_ENDED, iteration=iteration, text=text)
                break

        return state
