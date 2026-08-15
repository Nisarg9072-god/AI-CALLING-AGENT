"""
Unit tests — tool base, registry, and company service.
"""

from __future__ import annotations

import pytest

from app.agent.state import CallState, VerificationStatus
from app.company.repository import CompanyRepository
from app.company.service import CompanyService, ServiceError
from app.tools.base import BaseTool, ToolResult
from app.tools.customer import GetCustomerTool, VerifyCustomerTool
from app.tools.orders import GetOrderStatusTool, GetOrderTool
from app.tools.registry import ToolRegistry


# ── Tool base tests ────────────────────────────────────────────────────────────


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

    def execute(self, arguments, context):
        return ToolResult.ok({"echo": arguments["name"]}, tool_name=self.name)


class TestBaseTool:
    def test_validate_arguments_ok(self):
        tool = ConcreteTestTool()
        valid, err = tool.validate_arguments({"name": "test"})
        assert valid
        assert err == ""

    def test_validate_arguments_missing_required(self):
        tool = ConcreteTestTool()
        valid, err = tool.validate_arguments({"count": 5})
        assert not valid
        assert "name" in err

    def test_validate_arguments_wrong_type(self):
        tool = ConcreteTestTool()
        valid, err = tool.validate_arguments({"name": 123})  # should be string
        assert not valid
        assert "string" in err

    def test_tool_result_ok(self):
        result = ToolResult.ok({"key": "value"}, tool_name="test")
        assert result.success
        assert result.data["key"] == "value"
        assert result.error is None

    def test_tool_result_fail(self):
        result = ToolResult.fail("Something broke", tool_name="test")
        assert not result.success
        assert result.error == "Something broke"


# ── ToolRegistry tests ──────────────────────────────────────────────────────────


class TestToolRegistry:
    def _make_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(GetCustomerTool())
        registry.register(VerifyCustomerTool())
        return registry

    def test_register_and_get(self):
        registry = self._make_registry()
        tool = registry.get_tool("get_customer")
        assert tool is not None
        assert tool.name == "get_customer"

    def test_unknown_tool_returns_fail(self):
        registry = self._make_registry()
        state = CallState()
        result = registry.execute("nonexistent_tool", {}, state)
        assert not result.success
        assert "not registered" in (result.error or "")

    def test_get_customer_executes(self):
        registry = self._make_registry()
        state = CallState(customer_id="C001")
        result = registry.execute("get_customer", {"customer_id": "C001"}, state)
        assert result.success
        assert result.data["customer_id"] == "C001"
        assert result.data["name"] == "Alice Johnson"

    def test_missing_required_arg_blocked(self):
        registry = self._make_registry()
        state = CallState()
        result = registry.execute("get_customer", {}, state)  # Missing customer_id
        assert not result.success

    def test_tool_call_recorded_in_state(self):
        registry = self._make_registry()
        state = CallState(customer_id="C001")
        registry.execute("get_customer", {"customer_id": "C001"}, state)
        assert state.tool_call_count == 1

    def test_list_available_without_verification(self):
        registry = ToolRegistry(sensitive_tools=["sensitive_op"])

        class SensitiveTool(BaseTool):
            name = "sensitive_op"
            description = "Sensitive"
            required_permissions = ["verified"]
            def execute(self, a, c): return ToolResult.ok({})

        class PublicTool(BaseTool):
            name = "public_op"
            description = "Public"
            required_permissions = []
            def execute(self, a, c): return ToolResult.ok({})

        registry.register(SensitiveTool())
        registry.register(PublicTool())

        available_unverified = registry.list_available(is_verified=False)
        available_verified = registry.list_available(is_verified=True)

        assert "public_op" in available_unverified
        assert "sensitive_op" not in available_unverified
        assert "sensitive_op" in available_verified


# ── CompanyService tests ────────────────────────────────────────────────────────


class TestCompanyService:
    def _make_service(self) -> CompanyService:
        repo = CompanyRepository()
        return CompanyService(repo)

    def test_get_existing_customer(self):
        service = self._make_service()
        customer = service.get_customer("C001")
        assert customer["name"] == "Alice Johnson"
        assert "pin" not in customer  # PIN must never be returned

    def test_get_nonexistent_customer_raises(self):
        service = self._make_service()
        with pytest.raises(ServiceError):
            service.get_customer("INVALID")

    def test_verify_customer_correct_pin(self):
        service = self._make_service()
        assert service.verify_customer("C001", "1234") is True

    def test_verify_customer_wrong_pin(self):
        service = self._make_service()
        assert service.verify_customer("C001", "9999") is False

    def test_get_order(self):
        service = self._make_service()
        order = service.get_order("ORD-1001")
        assert order["order_id"] == "ORD-1001"
        assert order["status"] == "shipped"

    def test_get_orders_for_customer(self):
        service = self._make_service()
        orders = service.get_orders_for_customer("C001")
        assert len(orders) == 2
        order_ids = {o["order_id"] for o in orders}
        assert "ORD-1001" in order_ids

    def test_create_support_ticket(self):
        service = self._make_service()
        result = service.create_support_ticket("C001", "billing", "Test issue")
        assert result["status"] == "open"
        assert result["ticket_id"].startswith("TKT-")

    def test_schedule_callback(self):
        service = self._make_service()
        result = service.schedule_callback("C001", "tomorrow 2pm", "Follow-up")
        assert result["status"] == "scheduled"
        assert result["callback_id"].startswith("CB-")
