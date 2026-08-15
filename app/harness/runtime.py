"""
AgentRuntime — the top-level harness that wires everything together.

This is what the CLI (main.py) and the eval runner instantiate.
It owns: state, agent, tools, guardrails, trace logger, and the agent loop.
"""

from __future__ import annotations

from typing import Any

from app.agent.agent import Agent
from app.agent.llm_provider import LLMProvider, build_llm_provider
from app.agent.loop import AgentLoop
from app.agent.state import CallState, MessageRole, TerminationReason
from app.config import settings
from app.harness.guardrails import GuardrailEngine
from app.harness.policies import ALL_TOOL_NAMES, VERIFICATION_REQUIRED_TOOLS
from app.observability.trace import EventType, TraceLogger
from app.tools.calendar import ScheduleCallbackTool
from app.tools.customer import GetCustomerTool, VerifyCustomerTool
from app.tools.escalation import EndCallTool, TransferToHumanTool
from app.tools.orders import GetCustomerOrdersTool, GetOrderStatusTool, GetOrderTool
from app.tools.registry import ToolRegistry
from app.tools.support import CreateSupportTicketTool, GetCustomerTicketsTool


def _build_registry() -> ToolRegistry:
    """Instantiate all tools and register them."""
    registry = ToolRegistry(sensitive_tools=VERIFICATION_REQUIRED_TOOLS)
    for tool_cls in [
        GetCustomerTool,
        VerifyCustomerTool,
        GetOrderTool,
        GetOrderStatusTool,
        GetCustomerOrdersTool,
        ScheduleCallbackTool,
        CreateSupportTicketTool,
        GetCustomerTicketsTool,
        TransferToHumanTool,
        EndCallTool,
    ]:
        registry.register(tool_cls())
    return registry


class AgentRuntime:
    """
    Top-level harness for a single call session.

    Responsibilities:
    - Initialize call state
    - Wire up agent, tools, guardrails, trace logger, loop
    - Run the agent loop
    - Return final state + trace
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        customer_phone: str = "",
        trace_dir: str | None = None,
    ) -> None:
        self._llm = llm_provider or build_llm_provider()
        self._customer_phone = customer_phone
        self._trace_dir = trace_dir or settings.trace_dir

    def run(self, initial_message: str, customer_phone: str | None = None) -> tuple[CallState, TraceLogger]:
        """
        Execute a complete call session.

        Args:
            initial_message: The customer's first message (simulates STT output).
            customer_phone: Optional phone number override.

        Returns:
            (final_state, trace_logger) — inspect both for evaluation.
        """
        phone = customer_phone or self._customer_phone

        # ── Initialize state ──────────────────────────────────────────────────
        state = CallState(
            customer_phone=phone,
            max_iterations=settings.max_turns,
            max_tool_calls=settings.max_tool_calls,
        )
        state.available_tools = ALL_TOOL_NAMES

        # ── Initialize components ─────────────────────────────────────────────
        trace = TraceLogger(state.call_id, trace_dir=self._trace_dir)
        registry = _build_registry()
        agent = Agent(llm=self._llm, registry=registry)
        
        loop = AgentLoop(agent, registry, on_event=trace)

        # ── Record call start ─────────────────────────────────────────────────
        trace.record(
            EventType.CALL_STARTED,
            call_id=state.call_id,
            phone=phone,
        )

        # ── Run loop ─────────────────────────────────────────────────────────
        try:
            state.latest_user_message = initial_message
            final_state = loop.run(state)
        except Exception as exc:
            trace.record(EventType.ERROR, error=str(exc))
            state.terminate(TerminationReason.ERROR, outcome=str(exc))
            final_state = state

        # ── Save trace ────────────────────────────────────────────────────────
        if settings.trace_to_file:
            trace.save_to_file()

        return final_state, trace
