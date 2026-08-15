"""
Unit tests — Phase 2: State, Observation, Decision models.

These are pure model tests — no LLM, no tools, no network required.
"""

from __future__ import annotations

import pytest

from app.agent.decision import ActionType, AgentDecision, ValidationResult
from app.agent.observation import Observation, ToolResultSummary, build_observation
from app.agent.state import (
    CallState,
    EscalationStatus,
    ExtractedEntity,
    MessageRole,
    TerminationReason,
    ToolCallRecord,
    VerificationStatus,
)


# ════════════════════════════════════════════════════════════════════════
# CallState Tests
# ════════════════════════════════════════════════════════════════════════


class TestCallState:
    def test_initial_defaults(self):
        state = CallState()
        assert not state.finished
        assert state.verification_status == VerificationStatus.UNVERIFIED
        assert state.tool_call_count == 0
        assert state.current_iteration == 0
        assert state.escalation_status == EscalationStatus.NONE

    def test_call_id_generated(self):
        s1, s2 = CallState(), CallState()
        assert s1.call_id != s2.call_id
        assert len(s1.call_id) == 36   # UUID4 format

    def test_add_message(self):
        state = CallState()
        state.add_message(MessageRole.USER, "Hello")
        state.add_message(MessageRole.ASSISTANT, "Hi there!")
        assert len(state.conversation_history) == 2
        assert state.conversation_history[0].role == MessageRole.USER
        assert state.conversation_history[1].content == "Hi there!"

    def test_is_verified_false_initially(self):
        assert not CallState().is_verified

    def test_is_verified_true_after_verification(self):
        state = CallState()
        state.verification_status = VerificationStatus.VERIFIED
        assert state.is_verified

    def test_terminate(self):
        state = CallState()
        state.terminate(TerminationReason.AGENT_END, outcome="resolved")
        assert state.finished
        assert state.termination_reason == TerminationReason.AGENT_END
        assert state.outcome == "resolved"

    def test_tool_call_count(self):
        state = CallState()
        record = ToolCallRecord(
            idempotency_key="key-001",
            tool_name="get_order",
            arguments={"order_id": "ORD-1001"},
            result={"status": "shipped"},
            success=True,
            iteration=1,
        )
        state.record_tool_call(record)
        assert state.tool_call_count == 1
        assert state.tool_calls_remaining == state.max_tool_calls - 1

    def test_idempotency_check(self):
        state = CallState()
        record = ToolCallRecord(
            idempotency_key="idem-abc",
            tool_name="schedule_callback",
            arguments={},
            result={},
            success=True,
            iteration=1,
        )
        state.record_tool_call(record)
        assert state.is_idempotent_duplicate("idem-abc")
        assert not state.is_idempotent_duplicate("idem-xyz")

    def test_last_tool_result(self):
        state = CallState()
        assert state.last_tool_result is None
        state.record_tool_call(ToolCallRecord(
            idempotency_key="k1",
            tool_name="get_order",
            arguments={"order_id": "ORD-1"},
            result={"status": "delayed"},
            success=True,
            iteration=1,
        ))
        assert state.last_tool_result == {"status": "delayed"}
        assert state.last_tool_name == "get_order"

    def test_recent_messages_limit(self):
        state = CallState()
        for i in range(15):
            state.add_message(MessageRole.USER, f"Message {i}")
        recent = state.recent_messages(5)
        assert len(recent) == 5
        assert recent[-1].content == "Message 14"

    def test_turns_remaining(self):
        state = CallState(max_iterations=20)
        state.current_iteration = 5
        assert state.turns_remaining == 15

    def test_is_at_limit_false(self):
        state = CallState(max_iterations=20)
        state.current_iteration = 19
        assert not state.is_at_limit

    def test_is_at_limit_true(self):
        state = CallState(max_iterations=20)
        state.current_iteration = 20
        assert state.is_at_limit

    def test_get_entity(self):
        state = CallState()
        state.extracted_entities.append(
            ExtractedEntity(entity_type="order_id", value="ORD-1001")
        )
        assert state.get_entity("order_id") == "ORD-1001"
        assert state.get_entity("customer_id") is None

    def test_state_is_serializable(self):
        state = CallState(customer_id="C001", phone_number="+919537266092")
        data = state.model_dump()
        assert data["customer_id"] == "C001"
        assert data["phone_number"] == "+919537266092"
        # Round-trip
        state2 = CallState(**data)
        assert state2.call_id == state.call_id


# ════════════════════════════════════════════════════════════════════════
# AgentDecision Tests
# ════════════════════════════════════════════════════════════════════════


class TestAgentDecision:
    def test_speak_valid(self):
        d = AgentDecision(
            action=ActionType.SPEAK,
            response_text="Your order is delayed.",
            reasoning_summary="Tool returned delayed status.",
        )
        assert d.action == ActionType.SPEAK
        assert d.response_text == "Your order is delayed."

    def test_speak_missing_response_text_raises(self):
        with pytest.raises(ValueError, match="response_text is required"):
            AgentDecision(action=ActionType.SPEAK, reasoning_summary="test")

    def test_tool_call_valid(self):
        d = AgentDecision(
            action=ActionType.TOOL_CALL,
            tool_name="get_order_status",
            arguments={"order_id": "ORD-1001"},
            reasoning_summary="Customer asked about ORD-1001.",
        )
        assert d.tool_name == "get_order_status"
        assert d.arguments == {"order_id": "ORD-1001"}

    def test_tool_call_missing_tool_name_raises(self):
        with pytest.raises(ValueError, match="tool_name is required"):
            AgentDecision(action=ActionType.TOOL_CALL, reasoning_summary="test")

    def test_ask_clarification_valid(self):
        d = AgentDecision(
            action=ActionType.ASK_CLARIFICATION,
            response_text="Could you repeat your order number?",
            reasoning_summary="Order number was unclear.",
        )
        assert d.action == ActionType.ASK_CLARIFICATION

    def test_end_call_valid(self):
        d = AgentDecision(
            action=ActionType.END_CALL,
            response_text="Thank you! Have a great day.",
            reasoning_summary="Customer satisfied, issue resolved.",
        )
        assert d.action == ActionType.END_CALL

    def test_escalate_valid(self):
        d = AgentDecision(
            action=ActionType.ESCALATE,
            response_text="Let me transfer you to a specialist.",
            reasoning_summary="Customer requested human agent.",
        )
        assert d.action == ActionType.ESCALATE

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            AgentDecision(
                action=ActionType.SPEAK,
                response_text="Hi",
                reasoning_summary="test",
                confidence=1.5,   # > 1.0
            )
        with pytest.raises(ValueError):
            AgentDecision(
                action=ActionType.SPEAK,
                response_text="Hi",
                reasoning_summary="test",
                confidence=-0.1,  # < 0.0
            )

    def test_reasoning_summary_required(self):
        with pytest.raises(ValueError):
            AgentDecision(
                action=ActionType.SPEAK,
                response_text="Hi",
                # reasoning_summary missing
            )

    def test_decision_serializable(self):
        d = AgentDecision(
            action=ActionType.TOOL_CALL,
            tool_name="get_order",
            arguments={"order_id": "ORD-1"},
            reasoning_summary="Looking up order.",
        )
        data = d.model_dump()
        d2 = AgentDecision(**data)
        assert d2.tool_name == "get_order"


# ════════════════════════════════════════════════════════════════════════
# Observation Tests
# ════════════════════════════════════════════════════════════════════════


class TestObservation:
    def _state(self, **kwargs) -> CallState:
        s = CallState(**kwargs)
        s.available_tools = ["get_order", "end_call"]
        return s

    def test_build_observation_basic(self):
        state = self._state(customer_id="C001")
        state.current_user_message = "Where is my order?"
        obs = build_observation(state)
        assert obs.latest_user_message == "Where is my order?"
        assert obs.customer_id == "C001"
        assert obs.last_tool_result is None
        assert "get_order" in obs.available_tools

    def test_build_observation_with_tool_result(self):
        """The KEY agentic test: tool result flows into next observation."""
        state = self._state()
        state.current_user_message = "Can I schedule a callback?"
        tool_result = ToolResultSummary(
            tool_name="get_order_status",
            success=True,
            data={"status": "delayed", "order_id": "ORD-1001"},
        )
        obs = build_observation(state, last_tool_result=tool_result)
        assert obs.last_tool_result is not None
        assert obs.last_tool_result.tool_name == "get_order_status"
        assert obs.last_tool_result.data["status"] == "delayed"

    def test_build_observation_includes_recent_messages(self):
        state = self._state()
        state.add_message(MessageRole.USER, "Hello")
        state.add_message(MessageRole.ASSISTANT, "Hi!")
        obs = build_observation(state)
        assert len(obs.recent_messages) == 2
        assert obs.recent_messages[0]["role"] == "user"
        assert obs.recent_messages[1]["role"] == "assistant"

    def test_build_observation_limits_messages(self):
        state = self._state()
        for i in range(20):
            state.add_message(MessageRole.USER, f"msg {i}")
        obs = build_observation(state, recent_message_count=8)
        assert len(obs.recent_messages) == 8

    def test_build_observation_verification_reflected(self):
        state = self._state()
        state.verification_status = VerificationStatus.VERIFIED
        obs = build_observation(state)
        assert obs.is_verified is True

    def test_build_observation_limits_reflected(self):
        state = self._state(max_iterations=20, max_tool_calls=10)
        state.current_iteration = 5
        state.record_tool_call(ToolCallRecord(
            idempotency_key="k",
            tool_name="t",
            arguments={},
            result={},
            success=True,
            iteration=1,
        ))
        obs = build_observation(state)
        assert obs.turns_remaining == 15
        assert obs.tool_calls_remaining == 9

    def test_observation_serializable(self):
        state = self._state()
        state.current_user_message = "Test"
        obs = build_observation(state)
        data = obs.model_dump()
        obs2 = Observation(**data)
        assert obs2.latest_user_message == "Test"
