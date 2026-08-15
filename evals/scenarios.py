"""
Eval scenarios — 9 scripted test scenarios covering the full spec.

Each scenario defines:
- initial_message: What the customer says
- mock_responses: Scripted LLM decisions (deterministic, no API needed)
- expected_tools: Tools that MUST have been called
- forbidden_tools: Tools that must NOT have been called
- expected_termination: How the call should end
- min_turns: Minimum number of loop iterations expected
- description: Human-readable scenario name
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.agent.decision import ActionType, AgentDecision
from app.agent.state import TerminationReason


@dataclass
class EvalScenario:
    name: str
    description: str
    initial_message: str
    mock_responses: list[AgentDecision]
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_termination: Optional[TerminationReason] = None
    min_turns: int = 1
    customer_phone: str = "+1-555-0101"


# ── Scenario definitions ───────────────────────────────────────────────────────


def get_all_scenarios() -> list[EvalScenario]:
    return [
        # ── S1: Order status lookup ────────────────────────────────────────────
        EvalScenario(
            name="s1_order_status",
            description="Customer asks about order status — 3-step loop",
            initial_message="Hi, where is my order ORD-1001?",
            mock_responses=[
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="get_order_status",
                    arguments={"order_id": "ORD-1001"},
                    reasoning_summary="Customer asked for order status. Looking up ORD-1001.",
                    confidence=0.95,
                ),
                AgentDecision(
                    action=ActionType.SPEAK,
                    response_text="Your order ORD-1001 is currently shipped with tracking TRK-ALPHA-7823. Expected delivery in 2 days.",
                    reasoning_summary="Got order status — reporting to customer.",
                    confidence=0.98,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="end_call",
                    arguments={"outcome": "resolved", "summary": "Provided order status for ORD-1001"},
                    reasoning_summary="Issue resolved. Ending call.",
                    confidence=1.0,
                ),
            ],
            expected_tools=["get_order_status", "end_call"],
            expected_termination=TerminationReason.AGENT_END,
            min_turns=2,
        ),

        # ── S2: Customer lookup + multi-step ──────────────────────────────────
        EvalScenario(
            name="s2_customer_lookup_multi_step",
            description="Agent looks up customer then checks their orders — 4 steps",
            initial_message="I'd like to know about my recent orders",
            mock_responses=[
                AgentDecision(
                    action=ActionType.ASK_CLARIFICATION,
                    response_text="I'd be happy to help! Could you please provide your customer ID?",
                    reasoning_summary="Need customer ID to look up orders.",
                    confidence=0.9,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="get_customer",
                    arguments={"customer_id": "C001"},
                    reasoning_summary="Customer provided ID C001. Looking up their account.",
                    confidence=0.95,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="get_customer_orders",
                    arguments={"customer_id": "C001"},
                    reasoning_summary="Got customer. Now fetching their orders.",
                    confidence=0.97,
                ),
                AgentDecision(
                    action=ActionType.SPEAK,
                    response_text="You have 2 orders: ORD-1001 (shipped) and ORD-1002 (delivered).",
                    reasoning_summary="Summarizing order list for customer.",
                    confidence=0.95,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="end_call",
                    arguments={"outcome": "resolved", "summary": "Listed customer orders"},
                    reasoning_summary="Customer's question answered.",
                    confidence=1.0,
                ),
            ],
            expected_tools=["get_customer", "get_customer_orders", "end_call"],
            expected_termination=TerminationReason.AGENT_END,
            min_turns=3,
        ),

        # ── S3: Support ticket creation ───────────────────────────────────────
        EvalScenario(
            name="s3_support_ticket",
            description="Customer reports a technical issue — ticket created",
            initial_message="My order ORD-2001 has been delayed for 5 days. This is unacceptable.",
            mock_responses=[
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="get_order_status",
                    arguments={"order_id": "ORD-2001"},
                    reasoning_summary="Checking order status to confirm delay.",
                    confidence=0.95,
                ),
                AgentDecision(
                    action=ActionType.SPEAK,
                    response_text="I'm sorry to hear that. I can see ORD-2001 is indeed delayed. Let me create a support ticket for you.",
                    reasoning_summary="Confirming the delay, will create a ticket.",
                    confidence=0.93,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="create_support_ticket",
                    arguments={
                        "customer_id": "C002",
                        "issue_type": "delivery",
                        "description": "Order ORD-2001 delayed 5+ days, customer requesting escalation",
                    },
                    reasoning_summary="Creating delivery support ticket.",
                    confidence=0.97,
                ),
                AgentDecision(
                    action=ActionType.SPEAK,
                    response_text="I've created ticket TKT-XXXX for your delivery issue. Our team will follow up within 24 hours.",
                    reasoning_summary="Ticket created. Informing customer.",
                    confidence=0.95,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="end_call",
                    arguments={"outcome": "resolved", "summary": "Created delivery support ticket"},
                    reasoning_summary="Issue handled via ticket. Ending call.",
                    confidence=1.0,
                ),
            ],
            expected_tools=["get_order_status", "create_support_ticket", "end_call"],
            expected_termination=TerminationReason.AGENT_END,
            min_turns=3,
        ),

        # ── S4: Callback scheduling ───────────────────────────────────────────
        EvalScenario(
            name="s4_callback_scheduling",
            description="Customer requests a callback",
            initial_message="Can someone call me back tomorrow at 2pm?",
            mock_responses=[
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="schedule_callback",
                    arguments={
                        "customer_id": "C001",
                        "scheduled_time": "tomorrow at 2pm",
                        "reason": "Customer requested callback",
                    },
                    reasoning_summary="Scheduling callback as requested.",
                    confidence=0.97,
                ),
                AgentDecision(
                    action=ActionType.SPEAK,
                    response_text="I've scheduled a callback for tomorrow at 2pm. Our team will call you back at that time.",
                    reasoning_summary="Callback scheduled. Confirming to customer.",
                    confidence=0.95,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="end_call",
                    arguments={"outcome": "resolved", "summary": "Callback scheduled for tomorrow 2pm"},
                    reasoning_summary="Request fulfilled. Ending call.",
                    confidence=1.0,
                ),
            ],
            expected_tools=["schedule_callback", "end_call"],
            expected_termination=TerminationReason.AGENT_END,
            min_turns=2,
        ),

        # ── S5: Human escalation ──────────────────────────────────────────────
        EvalScenario(
            name="s5_human_escalation",
            description="Customer demands human agent — escalation path",
            initial_message="I want to speak to a human agent right now!",
            mock_responses=[
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="transfer_to_human",
                    arguments={
                        "reason": "Customer explicitly requested human agent",
                        "summary": "Customer called in asking to speak with a human agent directly",
                        "priority": "high",
                    },
                    reasoning_summary="Customer explicitly wants human. Escalating immediately.",
                    confidence=1.0,
                ),
            ],
            expected_tools=["transfer_to_human"],
            forbidden_tools=["get_customer", "get_order"],
            expected_termination=TerminationReason.ESCALATED,
            min_turns=1,
        ),

        # ── S6: Guardrail — max turns ─────────────────────────────────────────
        EvalScenario(
            name="s6_max_turns_guardrail",
            description="Guardrail blocks loop after max_iterations",
            initial_message="Hello?",
            mock_responses=[
                # Keep asking clarification until max turns hits
                AgentDecision(
                    action=ActionType.ASK_CLARIFICATION,
                    response_text="Could you tell me more about your issue?",
                    reasoning_summary="Need more info.",
                    confidence=0.5,
                ),
            ] * 25,  # More than max_turns
            expected_termination=TerminationReason.MAX_TURNS_REACHED,
            min_turns=1,
        ),

        # ── S7: Verification required ──────────────────────────────────────────
        EvalScenario(
            name="s7_verify_before_sensitive",
            description="Agent verifies customer before sensitive data",
            initial_message="I need to update my delivery address",
            mock_responses=[
                AgentDecision(
                    action=ActionType.ASK_CLARIFICATION,
                    response_text="For security, I need to verify your identity first. What is your 4-digit PIN?",
                    reasoning_summary="Address update is sensitive — must verify first.",
                    confidence=0.98,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="verify_customer",
                    arguments={"customer_id": "C001", "pin": "1234"},
                    reasoning_summary="Customer provided PIN. Verifying.",
                    confidence=0.97,
                ),
                AgentDecision(
                    action=ActionType.SPEAK,
                    response_text="Identity verified! Unfortunately address changes aren't yet available. Let me create a support ticket.",
                    reasoning_summary="Verified. But no address update tool — creating ticket instead.",
                    confidence=0.85,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="create_support_ticket",
                    arguments={
                        "customer_id": "C001",
                        "issue_type": "general",
                        "description": "Customer wants to update delivery address",
                    },
                    reasoning_summary="Creating ticket for address change.",
                    confidence=0.9,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="end_call",
                    arguments={"outcome": "resolved", "summary": "Ticket created for address update"},
                    reasoning_summary="Handled as best possible. Ending call.",
                    confidence=1.0,
                ),
            ],
            expected_tools=["verify_customer", "create_support_ticket", "end_call"],
            expected_termination=TerminationReason.AGENT_END,
            min_turns=2,
        ),

        # ── S8: Tool result changes decision ──────────────────────────────────
        EvalScenario(
            name="s8_tool_result_changes_decision",
            description="Agent adapts its decision based on tool result (observe→decide→act→observe→decide)",
            initial_message="What's the status of order ORD-2001?",
            mock_responses=[
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="get_order_status",
                    arguments={"order_id": "ORD-2001"},
                    reasoning_summary="Customer asked about ORD-2001 status.",
                    confidence=0.97,
                ),
                # This decision changes BECAUSE the tool returned "delayed"
                AgentDecision(
                    action=ActionType.SPEAK,
                    response_text="I can see that ORD-2001 is currently delayed. I'm sorry for the inconvenience.",
                    reasoning_summary="Tool showed 'delayed' status — adapting response to address the delay.",
                    confidence=0.95,
                ),
                AgentDecision(
                    action=ActionType.ASK_CLARIFICATION,
                    response_text="Would you like me to create a support ticket or schedule a callback?",
                    reasoning_summary="Offering next steps given the delay.",
                    confidence=0.92,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="end_call",
                    arguments={"outcome": "resolved", "summary": "Explained delay, offered options"},
                    reasoning_summary="Customer acknowledged. Ending call.",
                    confidence=1.0,
                ),
            ],
            expected_tools=["get_order_status", "end_call"],
            expected_termination=TerminationReason.AGENT_END,
            min_turns=3,
        ),

        # ── S9: Multi-tool multi-step reasoning ───────────────────────────────
        EvalScenario(
            name="s9_multi_tool_reasoning",
            description="Complex multi-step: lookup customer + orders + ticket + end",
            initial_message="I'm customer C003. I placed an order but never got it.",
            mock_responses=[
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="get_customer",
                    arguments={"customer_id": "C003"},
                    reasoning_summary="Customer identified as C003. Looking up account.",
                    confidence=0.95,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="get_customer_orders",
                    arguments={"customer_id": "C003"},
                    reasoning_summary="Got customer info. Now checking their orders.",
                    confidence=0.95,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="get_order_status",
                    arguments={"order_id": "ORD-3001"},
                    reasoning_summary="Found order ORD-3001. Checking its status.",
                    confidence=0.93,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="create_support_ticket",
                    arguments={
                        "customer_id": "C003",
                        "issue_type": "delivery",
                        "description": "Customer C003 reports not receiving order ORD-3001 (status: processing)",
                    },
                    reasoning_summary="Order still processing but customer expects delivery. Creating ticket.",
                    confidence=0.9,
                ),
                AgentDecision(
                    action=ActionType.SPEAK,
                    response_text="I've looked into your order ORD-3001 and created a support ticket. Our team will follow up.",
                    reasoning_summary="Summarizing actions taken.",
                    confidence=0.95,
                ),
                AgentDecision(
                    action=ActionType.TOOL_CALL,
                    tool_name="end_call",
                    arguments={"outcome": "resolved", "summary": "Created delivery ticket for C003 order ORD-3001"},
                    reasoning_summary="All actions taken. Ending call.",
                    confidence=1.0,
                ),
            ],
            expected_tools=["get_customer", "get_customer_orders", "get_order_status", "create_support_ticket", "end_call"],
            expected_termination=TerminationReason.AGENT_END,
            min_turns=4,
        ),
    ]
