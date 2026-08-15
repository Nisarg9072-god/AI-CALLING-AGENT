"""
Unit tests — state model, decision schema, tool validation.
"""

from __future__ import annotations

import pytest

from app.agent.decision import ActionType, AgentDecision
from app.agent.state import (
    CallState,
    EscalationStatus,
    MessageRole,
    TerminationReason,
    ToolCallRecord,
    VerificationStatus,
)


# ── CallState tests ────────────────────────────────────────────────────────────


class TestCallState:
    def test_initial_state_not_finished(self):
        state = CallState()
        assert not state.finished

    def test_add_message(self):
        state = CallState()
        state.add_message(MessageRole.USER, "Hello")
        assert len(state.conversation_history) == 1
        assert state.conversation_history[0].role == MessageRole.USER
        assert state.conversation_history[0].content == "Hello"

    def test_is_verified_false_initially(self):
        state = CallState()
        assert not state.is_verified

    def test_is_verified_true_after_verification(self):
        state = CallState()
        state.verification_status = VerificationStatus.VERIFIED
        assert state.is_verified

    def test_terminate_sets_finished(self):
        state = CallState()
        state.terminate(TerminationReason.AGENT_END, outcome="resolved")
        assert state.finished
        assert state.termination_reason == TerminationReason.AGENT_END

    def test_tool_call_count(self):
        state = CallState()
        record = ToolCallRecord(
            idempotency_key="abc123",
            tool_name="get_order",
            arguments={"order_id": "ORD-1001"},
            result={"status": "shipped"},
            success=True,
            iteration=1,
        )
        state.record_tool_call(record)
        assert state.tool_call_count == 1

    def test_idempotency_check(self):
        state = CallState()
        record = ToolCallRecord(
            idempotency_key="idem-001",
            tool_name="get_order",
            arguments={},
            result={},
            success=True,
            iteration=1,
        )
        state.record_tool_call(record)
        assert state.is_idempotent_duplicate("idem-001")
        assert not state.is_idempotent_duplicate("idem-002")

    def test_recent_messages_limit(self):
        state = CallState()
        for i in range(15):
            state.add_message(MessageRole.USER, f"msg {i}")
        recent = state.recent_messages(5)
        assert len(recent) == 5
        assert recent[-1].content == "msg 14"


# ── AgentDecision tests ────────────────────────────────────────────────────────


class TestAgentDecision:
    def test_speak_requires_response_text(self):
        with pytest.raises(ValueError):
            AgentDecision(
                action=ActionType.SPEAK,
                reasoning_summary="test",
            )

    def test_tool_call_requires_tool_name(self):
        with pytest.raises(ValueError):
            AgentDecision(
                action=ActionType.TOOL_CALL,
                reasoning_summary="test",
            )

    def test_tool_call_valid(self):
        d = AgentDecision(
            action=ActionType.TOOL_CALL,
            tool_name="get_order",
            arguments={"order_id": "ORD-1001"},
            reasoning_summary="Looking up order",
        )
        assert d.action == ActionType.TOOL_CALL
        assert d.tool_name == "get_order"

    def test_speak_valid(self):
        d = AgentDecision(
            action=ActionType.SPEAK,
            response_text="Hello, how can I help?",
            reasoning_summary="Greeting customer",
        )
        assert d.response_text == "Hello, how can I help?"

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            AgentDecision(
                action=ActionType.SPEAK,
                response_text="Hi",
                reasoning_summary="test",
                confidence=1.5,  # > 1.0
            )

    def test_end_call_valid(self):
        d = AgentDecision(
            action=ActionType.END_CALL,
            response_text="Goodbye!",
            reasoning_summary="Call resolved",
        )
        assert d.action == ActionType.END_CALL
