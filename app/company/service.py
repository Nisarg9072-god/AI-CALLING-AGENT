"""
CompanyService — business logic layer.

This is the ONLY gateway the tools use to access company data.
The agent never calls the repository directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.company.repository import (
    Callback,
    CompanyRepository,
    Customer,
    Order,
    SupportTicket,
    get_repository,
)


class ServiceError(Exception):
    """Raised when a business-logic rule is violated."""


class CompanyService:
    def __init__(self, repo: CompanyRepository | None = None) -> None:
        self._repo = repo or get_repository()

    # ── Customer ───────────────────────────────────────────────────────────────

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        customer = self._repo.get_customer(customer_id)
        if not customer:
            raise ServiceError(f"Customer '{customer_id}' not found")
        return self._customer_to_dict(customer)

    def find_customer_by_phone(self, phone: str) -> dict[str, Any] | None:
        customer = self._repo.find_customer_by_phone(phone)
        if not customer:
            return None
        return self._customer_to_dict(customer)

    def verify_customer(self, customer_id: str, pin: str) -> bool:
        """Deterministic PIN check — never probabilistic."""
        customer = self._repo.get_customer(customer_id)
        if not customer:
            raise ServiceError(f"Customer '{customer_id}' not found")
        return customer.pin == pin.strip()

    # ── Orders ─────────────────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> dict[str, Any]:
        order = self._repo.get_order(order_id)
        if not order:
            raise ServiceError(f"Order '{order_id}' not found")
        return self._order_to_dict(order)

    def get_orders_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        orders = self._repo.get_orders_for_customer(customer_id)
        return [self._order_to_dict(o) for o in orders]

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        order = self._repo.get_order(order_id)
        if not order:
            raise ServiceError(f"Order '{order_id}' not found")
        return {
            "order_id": order.order_id,
            "status": order.status,
            "estimated_delivery": order.estimated_delivery,
            "tracking_number": order.tracking_number,
        }

    def schedule_callback(
        self, customer_id: str, scheduled_time: str, reason: str
    ) -> dict[str, Any]:
        callback_id = f"CB-{uuid.uuid4().hex[:6].upper()}"
        callback = Callback(
            callback_id=callback_id,
            customer_id=customer_id,
            scheduled_time=scheduled_time,
            reason=reason,
        )
        created = self._repo.create_callback(callback)
        return {
            "callback_id": created.callback_id,
            "scheduled_time": created.scheduled_time,
            "status": created.status,
            "message": "Callback scheduled successfully.",
        }

    # ── Support tickets ────────────────────────────────────────────────────────

    def create_support_ticket(
        self, customer_id: str, issue_type: str, description: str
    ) -> dict[str, Any]:
        ticket_id = f"TKT-{uuid.uuid4().hex[:4].upper()}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ticket = SupportTicket(
            ticket_id=ticket_id,
            customer_id=customer_id,
            issue_type=issue_type,
            description=description,
            status="open",
            created_at=now,
        )
        created = self._repo.create_ticket(ticket)
        return {
            "ticket_id": created.ticket_id,
            "status": created.status,
            "message": f"Support ticket {created.ticket_id} created successfully.",
        }

    def get_tickets_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        tickets = self._repo.get_tickets_for_customer(customer_id)
        return [self._ticket_to_dict(t) for t in tickets]

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _customer_to_dict(c: Customer) -> dict[str, Any]:
        return {
            "customer_id": c.customer_id,
            "name": c.name,
            "phone": c.phone,
            "email": c.email,
            "account_status": c.account_status,
            # PIN never returned
        }

    @staticmethod
    def _order_to_dict(o: Order) -> dict[str, Any]:
        return {
            "order_id": o.order_id,
            "customer_id": o.customer_id,
            "status": o.status,
            "items": o.items,
            "total_amount": o.total_amount,
            "delivery_address": o.delivery_address,
            "estimated_delivery": o.estimated_delivery,
            "tracking_number": o.tracking_number,
            "placed_at": o.placed_at,
        }

    @staticmethod
    def _ticket_to_dict(t: SupportTicket) -> dict[str, Any]:
        return {
            "ticket_id": t.ticket_id,
            "customer_id": t.customer_id,
            "issue_type": t.issue_type,
            "description": t.description,
            "status": t.status,
            "created_at": t.created_at,
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_service: CompanyService | None = None


def get_service() -> CompanyService:
    global _service
    if _service is None:
        _service = CompanyService()
    return _service
