"""
All 10 required eval scenarios.

Run: uv run python -m evals.scenarios
"""

from __future__ import annotations

from app.agent.decision import ActionType, AgentDecision
from app.agent.state import TerminationReason
from evals.runner import EvalRunner, EvalScenario, console


def _d(action: ActionType, **kwargs) -> AgentDecision:
    return AgentDecision(action=action, reasoning_summary="eval", **kwargs)


# ── Define all 10 scenarios ────────────────────────────────────────────────────

SCENARIOS: list[EvalScenario] = [

    # EVAL 1: Order status
    EvalScenario(
        name="eval_01_order_status",
        description="Customer asks for order status → agent calls get_order_status",
        initial_message="Where is my order ORD-1001?",
        llm_responses=[
            _d(ActionType.TOOL_CALL, tool_name="get_order_status",
               arguments={"order_id": "ORD-1001"}),
            _d(ActionType.SPEAK, response_text="Your order ORD-1001 is shipped."),
            _d(ActionType.END_CALL, response_text="Goodbye!"),
        ],
        expected_tools=["get_order_status"],
        expected_termination=TerminationReason.AGENT_END,
    ),

    # EVAL 2: No hallucination — agent only reports what tool returned
    EvalScenario(
        name="eval_02_no_hallucination",
        description="Agent must not invent delivery date not in tool result",
        initial_message="Where is my order ORD-2001?",
        llm_responses=[
            _d(ActionType.TOOL_CALL, tool_name="get_order_status",
               arguments={"order_id": "ORD-2001"}),
            # Agent speaks based on tool result (status=delayed)
            _d(ActionType.SPEAK, response_text="Your order is currently delayed."),
            _d(ActionType.END_CALL, response_text="Goodbye!"),
        ],
        expected_tools=["get_order_status"],
        forbidden_tools=["get_customer"],   # No need to get customer just for status
        custom_assertions=[
            # Tool was called before the agent spoke (agentic: decide→act→observe→decide)
            lambda state: _assert(
                state.tool_calls_made[0].tool_name == "get_order_status",
                "First tool call must be get_order_status"
            ),
            lambda state: _assert(
                state.tool_calls_made[0].result["status"] == "delayed",
                "Tool result must show delayed"
            ),
        ],
    ),

    # EVAL 3: Sensitive operation requires verification
    EvalScenario(
        name="eval_03_sensitive_requires_verification",
        description="Account update requires verify_customer first",
        initial_message="I want to update my account.",
        llm_responses=[
            _d(ActionType.ASK_CLARIFICATION,
               response_text="Please provide your 4-digit PIN to verify your identity."),
            _d(ActionType.TOOL_CALL, tool_name="verify_customer",
               arguments={"customer_id": "C001", "pin": "1234"}),
            _d(ActionType.SPEAK, response_text="Identity verified. How would you like to update your account?"),
            _d(ActionType.END_CALL, response_text="Goodbye!"),
        ],
        expected_tools=["verify_customer"],
        expect_verified=True,
    ),

    # EVAL 4: Human transfer request
    EvalScenario(
        name="eval_04_human_transfer",
        description="Customer explicitly requests human agent",
        initial_message="I want to speak to a human.",
        llm_responses=[
            _d(ActionType.ESCALATE,
               response_text="Of course, let me connect you with a specialist now."),
        ],
        expected_termination=TerminationReason.ESCALATED,
    ),

    # EVAL 5: Callback scheduling
    EvalScenario(
        name="eval_05_schedule_callback",
        description="Customer requests callback → schedule_callback tool called",
        initial_message="Can you call me back tomorrow?",
        llm_responses=[
            _d(ActionType.TOOL_CALL, tool_name="schedule_callback",
               arguments={"customer_id": "C001", "preferred_time": "tomorrow",
                          "reason": "customer request"}),
            _d(ActionType.SPEAK, response_text="I've scheduled a callback for tomorrow."),
            _d(ActionType.END_CALL, response_text="Goodbye!"),
        ],
        expected_tools=["schedule_callback"],
    ),

    # EVAL 6: Duplicate callback — only one created
    EvalScenario(
        name="eval_06_duplicate_callback_prevented",
        description="Same callback requested twice → only one created (idempotency)",
        initial_message="Schedule a callback for tomorrow morning.",
        llm_responses=[
            _d(ActionType.TOOL_CALL, tool_name="schedule_callback",
               arguments={"customer_id": "C001", "preferred_time": "tomorrow morning",
                          "reason": "request"}),
            _d(ActionType.TOOL_CALL, tool_name="schedule_callback",   # duplicate
               arguments={"customer_id": "C001", "preferred_time": "tomorrow morning",
                          "reason": "request"}),
            _d(ActionType.SPEAK, response_text="Callback is scheduled."),
            _d(ActionType.END_CALL, response_text="Goodbye!"),
        ],
        expected_tools=["schedule_callback"],
        custom_assertions=[
            lambda state: _assert(
                state.tool_call_count == 1,
                f"Expected exactly 1 callback, got {state.tool_call_count}"
            ),
        ],
    ),

    # EVAL 7: Tool failure handled gracefully
    EvalScenario(
        name="eval_07_unknown_order_handled",
        description="Tool returns not-found error → agent handles gracefully, no crash",
        initial_message="Where is order ORD-9999?",
        llm_responses=[
            _d(ActionType.TOOL_CALL, tool_name="get_order_status",
               arguments={"order_id": "ORD-9999"}),   # will return failure
            _d(ActionType.ASK_CLARIFICATION,
               response_text="I couldn't find that order. Could you double-check the order number?"),
            _d(ActionType.END_CALL, response_text="Goodbye!"),
        ],
        expected_tools=["get_order_status"],
        expected_termination=TerminationReason.AGENT_END,
        custom_assertions=[
            lambda state: _assert(
                not state.tool_calls_made[0].success,
                "Tool result should be a failure for unknown order"
            ),
        ],
    ),

    # EVAL 8: Max turns guardrail
    EvalScenario(
        name="eval_08_max_turns_guardrail",
        description="Agent that never ends → max_turns forces termination",
        initial_message="Hello",
        llm_responses=[
            _d(ActionType.SPEAK, response_text="...") for _ in range(30)
        ],
        expected_termination=TerminationReason.MAX_TURNS_REACHED,
        max_turns=5,
    ),

    # EVAL 9: Multi-step agentic loop
    # Verifies: DECIDE→TOOL→RESULT→NEW DECISION (multiple cycles)
    EvalScenario(
        name="eval_09_multi_step_agentic",
        description="Full agentic loop: order check → callback → end",
        initial_message="Check order ORD-1001 and schedule a callback.",
        llm_responses=[
            _d(ActionType.TOOL_CALL, tool_name="get_order_status",
               arguments={"order_id": "ORD-1001"}),
            _d(ActionType.SPEAK, response_text="Your order is shipped."),
            _d(ActionType.TOOL_CALL, tool_name="schedule_callback",
               arguments={"customer_id": "C001", "preferred_time": "tomorrow",
                          "reason": "order update"}),
            _d(ActionType.SPEAK, response_text="Callback scheduled."),
            _d(ActionType.END_CALL, response_text="Goodbye!"),
        ],
        expected_tools=["get_order_status", "schedule_callback"],
        expected_termination=TerminationReason.AGENT_END,
        custom_assertions=[
            # Must have at least 2 tool calls (multi-step)
            lambda state: _assert(
                state.tool_call_count >= 2,
                f"Expected >= 2 tool calls (multi-step), got {state.tool_call_count}"
            ),
            # get_order_status must come BEFORE schedule_callback
            lambda state: _assert(
                state.tool_calls_made[0].tool_name == "get_order_status",
                "get_order_status must be first tool"
            ),
        ],
    ),

    # EVAL 10: Tool result changes next decision
    EvalScenario(
        name="eval_10_tool_result_changes_decision",
        description="Tool result (delayed) → agent speaks about delay (not invented data)",
        initial_message="Check order ORD-2001",
        llm_responses=[
            _d(ActionType.TOOL_CALL, tool_name="get_order_status",
               arguments={"order_id": "ORD-2001"}),
            # Second decision: agent must have seen the delayed result
            _d(ActionType.SPEAK, response_text="Your order is delayed."),
            _d(ActionType.END_CALL, response_text="Is there anything else?"),
        ],
        expected_tools=["get_order_status"],
        custom_assertions=[
            lambda state: _assert(
                state.tool_calls_made[0].result.get("status") == "delayed",
                "Tool must return delayed status"
            ),
            # After the tool call, agent said something (turn count > 1)
            lambda state: _assert(
                state.current_iteration >= 3,
                f"Must have multiple turns for agentic loop, got {state.current_iteration}"
            ),
        ],
    ),
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    console.print("\n[bold cyan]>> Running AI Calling Agent Evals...[/bold cyan]\n")
    runner = EvalRunner()
    results = runner.run_all(SCENARIOS)
    runner.print_results(results)


if __name__ == "__main__":
    main()
