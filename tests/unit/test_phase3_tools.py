"""
Unit tests — Phase 3: Tool abstraction, ToolRegistry, and all tools.

No LLM, no network. All tests are deterministic.
"""

from __future__ import annotations

import pytest

from app.agent.state import CallState, VerificationStatus
from app.tools.base import BaseTool, ToolContext, ToolResult
from app.tools.calendar import ScheduleCallbackTool
from app.tools.customer import GetCustomerTool, VerifyCustomerTool
from app.tools.escalation import EndCallTool, TransferToHumanTool
from app.tools.factory import build_tool_registry
from app.tools.orders import GetOrderStatusTool, GetOrderTool
from app.tools.registry import ToolRegistry, _make_idempotency_key
from app.tools.support import CreateSupportTicketTool


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ctx(**kwargs) -> ToolContext:
    return ToolContext(call_id="test-call", customer_id="C001", **kwargs)


def _state(is_verified: bool = False) -> CallState:
    s = CallState(customer_id="C001")
    if is_verified:
        s.verification_status = VerificationStatus.VERIFIED
    return s


# ════════════════════════════════════════════════════════════════════════
# BaseTool Validation
# ════════════════════════════════════════════════════════════════════════


class ConcreteTestTool(BaseTool):
    name = "test_tool"
    description = "A test tool"
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["name"],
    }
    required_permissions: list[str] = []

    def execute(self, arguments, context):
        return ToolResult.ok({"echo": arguments["name"]}, tool_name=self.name)


class TestBaseToolValidation:
    def test_valid_arguments(self):
        tool = ConcreteTestTool()
        ok, msg = tool.validate_arguments({"name": "hello"})
        assert ok
        assert msg == ""

    def test_missing_required_field(self):
        tool = ConcreteTestTool()
        ok, msg = tool.validate_arguments({})
        assert not ok
        assert "name" in msg

    def test_wrong_type_string_for_integer(self):
        tool = ConcreteTestTool()
        ok, msg = tool.validate_arguments({"name": "hi", "count": "five"})
        assert not ok
        assert "count" in msg

    def test_extra_fields_allowed(self):
        tool = ConcreteTestTool()
        ok, _ = tool.validate_arguments({"name": "hi", "extra": "allowed"})
        assert ok

    def test_bool_not_integer(self):
        tool = ConcreteTestTool()
        ok, msg = tool.validate_arguments({"name": "hi", "count": True})
        assert not ok   # bool is NOT accepted as integer

    def test_tool_result_ok(self):
        r = ToolResult.ok({"status": "shipped"}, tool_name="get_order")
        assert r.success
        assert r.data["status"] == "shipped"
        assert r.error is None

    def test_tool_result_fail(self):
        r = ToolResult.fail("Not found", tool_name="get_order")
        assert not r.success
        assert r.error == "Not found"
        assert r.data == {}


# ════════════════════════════════════════════════════════════════════════
# ToolRegistry
# ════════════════════════════════════════════════════════════════════════


class TestToolRegistry:
    def _registry(self, sensitive: list[str] | None = None) -> ToolRegistry:
        r = ToolRegistry(sensitive_tools=sensitive or [])
        r.register(ConcreteTestTool())
        return r

    def test_register_and_lookup(self):
        r = self._registry()
        assert "test_tool" in r
        assert r.get_tool("test_tool") is not None

    def test_duplicate_registration_raises(self):
        r = self._registry()
        with pytest.raises(ValueError, match="already registered"):
            r.register(ConcreteTestTool())

    def test_unnamed_tool_raises(self):
        class BadTool(BaseTool):
            name = ""
            def execute(self, a, c): ...
        r = ToolRegistry()
        with pytest.raises(ValueError, match="no name"):
            r.register(BadTool())

    def test_unknown_tool_returns_fail(self):
        r = self._registry()
        result = r.execute("ghost_tool", {}, _state())
        assert not result.success
        assert "not registered" in result.error

    def test_execute_valid_tool(self):
        r = self._registry()
        result = r.execute("test_tool", {"name": "alice"}, _state())
        assert result.success
        assert result.data["echo"] == "alice"

    def test_schema_validation_blocks_bad_args(self):
        r = self._registry()
        result = r.execute("test_tool", {}, _state())   # missing 'name'
        assert not result.success
        assert "Missing required" in result.error

    def test_execution_recorded_in_state(self):
        r = self._registry()
        state = _state()
        r.execute("test_tool", {"name": "bob"}, state)
        assert state.tool_call_count == 1
        assert state.tool_calls_made[0].tool_name == "test_tool"

    def test_idempotency_prevents_duplicate(self):
        r = self._registry()
        state = _state()
        r.execute("test_tool", {"name": "carol"}, state)
        r.execute("test_tool", {"name": "carol"}, state)
        assert state.tool_call_count == 1   # second call is idempotent

    def test_idempotent_result_has_flag(self):
        r = self._registry()
        state = _state()
        r.execute("test_tool", {"name": "carol"}, state)
        result2 = r.execute("test_tool", {"name": "carol"}, state)
        assert result2.data.get("_idempotent") is True

    def test_sensitive_tool_blocked_without_verification(self):
        """Tools marked sensitive cannot run for unverified customers."""
        r = ToolRegistry(sensitive_tools=["test_tool"])
        r.register(ConcreteTestTool())
        result = r.execute("test_tool", {"name": "x"}, _state(is_verified=False))
        assert not result.success
        assert "verification" in result.error.lower()

    def test_sensitive_tool_allowed_with_verification(self):
        r = ToolRegistry(sensitive_tools=["test_tool"])
        r.register(ConcreteTestTool())
        result = r.execute("test_tool", {"name": "x"}, _state(is_verified=True))
        assert result.success

    def test_list_available_excludes_sensitive_unverified(self):
        r = ToolRegistry(sensitive_tools=["test_tool"])
        r.register(ConcreteTestTool())
        available = r.list_available(is_verified=False)
        assert "test_tool" not in available

    def test_list_available_includes_sensitive_when_verified(self):
        r = ToolRegistry(sensitive_tools=["test_tool"])
        r.register(ConcreteTestTool())
        available = r.list_available(is_verified=True)
        assert "test_tool" in available

    def test_idempotency_key_stable(self):
        key1 = _make_idempotency_key("get_order", {"order_id": "ORD-1"})
        key2 = _make_idempotency_key("get_order", {"order_id": "ORD-1"})
        assert key1 == key2

    def test_idempotency_key_differs_for_different_args(self):
        key1 = _make_idempotency_key("get_order", {"order_id": "ORD-1"})
        key2 = _make_idempotency_key("get_order", {"order_id": "ORD-2"})
        assert key1 != key2


# ════════════════════════════════════════════════════════════════════════
# Individual Tool Tests
# ════════════════════════════════════════════════════════════════════════


class TestGetCustomerTool:
    def test_existing_customer(self):
        result = GetCustomerTool().execute({"customer_id": "C001"}, _ctx())
        assert result.success
        assert result.data["name"] == "Alice Johnson"
        assert "pin_hash" not in result.data   # PIN never returned

    def test_missing_customer(self):
        result = GetCustomerTool().execute({"customer_id": "XXXX"}, _ctx())
        assert not result.success

    def test_tool_name_set(self):
        result = GetCustomerTool().execute({"customer_id": "C001"}, _ctx())
        assert result.tool_name == "get_customer"


class TestVerifyCustomerTool:
    def test_correct_pin(self):
        result = VerifyCustomerTool().execute({"customer_id": "C001", "pin": "1234"}, _ctx())
        assert result.success
        assert result.data["verified"] is True

    def test_wrong_pin(self):
        result = VerifyCustomerTool().execute({"customer_id": "C001", "pin": "0000"}, _ctx())
        assert result.success
        assert result.data["verified"] is False  # not a tool error, just wrong PIN

    def test_missing_customer(self):
        result = VerifyCustomerTool().execute({"customer_id": "XXXX", "pin": "1234"}, _ctx())
        assert not result.success


class TestGetOrderStatusTool:
    def test_shipped_order(self):
        result = GetOrderStatusTool().execute({"order_id": "ORD-1001"}, _ctx())
        assert result.success
        assert result.data["status"] == "shipped"
        assert "estimated_delivery" in result.data

    def test_delayed_order(self):
        result = GetOrderStatusTool().execute({"order_id": "ORD-2001"}, _ctx())
        assert result.success
        assert result.data["status"] == "delayed"

    def test_missing_order(self):
        result = GetOrderStatusTool().execute({"order_id": "ORD-9999"}, _ctx())
        assert not result.success


class TestScheduleCallbackTool:
    def test_creates_callback(self):
        result = ScheduleCallbackTool().execute(
            {"customer_id": "C001", "preferred_time": "tomorrow 2pm", "reason": "order delay"},
            _ctx(),
        )
        assert result.success
        assert "callback_id" in result.data
        assert result.data["status"] == "scheduled"

    def test_idempotent_callback_not_duplicated(self):
        """Two calls with the same args at tool layer → same callback."""
        args = {"customer_id": "C001", "preferred_time": "2026-08-17T10:00", "reason": "test"}
        t = ScheduleCallbackTool()
        r1 = t.execute(args, _ctx())
        r2 = t.execute(args, _ctx())
        assert r1.data["callback_id"] == r2.data["callback_id"]


class TestTransferToHumanTool:
    def test_transfer(self):
        result = TransferToHumanTool().execute(
            {"reason": "customer request", "summary": "Order delay issue"},
            _ctx(),
        )
        assert result.success
        assert result.data["transferred"] is True

    def test_priority_defaults_to_normal(self):
        result = TransferToHumanTool().execute(
            {"reason": "test", "summary": "test"},
            _ctx(),
        )
        assert result.data["priority"] == "normal"


class TestEndCallTool:
    def test_end_call(self):
        result = EndCallTool().execute(
            {"outcome": "resolved", "summary": "Customer satisfied"},
            _ctx(),
        )
        assert result.success
        assert result.data["ended"] is True
        assert result.data["outcome"] == "resolved"


# ════════════════════════════════════════════════════════════════════════
# Full Registry (build_tool_registry)
# ════════════════════════════════════════════════════════════════════════


class TestBuildToolRegistry:
    def test_all_tools_registered(self):
        r = build_tool_registry()
        expected = [
            "get_customer", "verify_customer",
            "get_order", "get_order_status", "get_customer_orders",
            "create_support_ticket", "get_customer_tickets",
            "schedule_callback",
            "transfer_to_human", "end_call",
        ]
        for name in expected:
            assert name in r, f"'{name}' not registered"

    def test_tool_count(self):
        r = build_tool_registry()
        assert len(r) == 10

    def test_end_to_end_order_status_via_registry(self):
        """Full pipeline: registry → validation → tool → result recorded in state."""
        r = build_tool_registry()
        state = _state()
        result = r.execute("get_order_status", {"order_id": "ORD-1001"}, state)
        assert result.success
        assert result.data["status"] == "shipped"
        assert state.tool_call_count == 1
        # State captures the result for next observation
        assert state.last_tool_result == result.data

    def test_create_support_ticket_via_registry(self):
        r = build_tool_registry()
        state = _state()
        result = r.execute(
            "create_support_ticket",
            {"customer_id": "C001", "category": "shipping", "description": "Order delayed"},
            state,
        )
        assert result.success
        assert result.data["ticket_id"].startswith("TKT-")
