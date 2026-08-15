"""
Business logic layer — the ONLY gateway to company data.

This uses SQLAlchemy repositories.
Access path: Agent → Tool → CompanyService → Repository → Database
"""

from __future__ import annotations

from typing import Any

from app.database import SessionLocal
from app.company.repository import (
    CustomerRepository,
    OrderRepository,
    TicketRepository,
    CallbackRepository,
)


class CompanyServiceError(Exception):
    """Raised when a business rule is violated or data is not found."""


class CompanyService:
    """
    Business logic layer.
    Manages DB sessions per method call to stay thread-safe and stateless.
    """

    # ── Customer ──────────────────────────────────────────────────────────────

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Return customer data (never includes PIN)."""
        with SessionLocal() as db:
            repo = CustomerRepository(db)
            customer = repo.get_by_id(customer_id)
            if not customer:
                raise CompanyServiceError(f"Customer '{customer_id}' not found.")
            return {k: v for k, v in customer.items() if k != "pin_hash"}

    def verify_customer(self, customer_id: str, pin: str) -> bool:
        """Verify customer identity. Returns True if PIN matches."""
        with SessionLocal() as db:
            repo = CustomerRepository(db)
            customer = repo.get_by_id(customer_id)
            if not customer:
                raise CompanyServiceError(f"Customer '{customer_id}' not found.")
            return customer.get("pin_hash") == pin

    def find_customer_by_phone(self, phone: str) -> dict[str, Any] | None:
        """Look up a customer by phone number."""
        with SessionLocal() as db:
            repo = CustomerRepository(db)
            customer = repo.get_by_phone(phone)
            if not customer:
                return None
            return {k: v for k, v in customer.items() if k != "pin_hash"}

    # ── Orders ────────────────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> dict[str, Any]:
        with SessionLocal() as db:
            repo = OrderRepository(db)
            order = repo.get_by_id(order_id)
            if not order:
                raise CompanyServiceError(f"Order '{order_id}' not found.")
            return order

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Return status-only subset of order data."""
        order = self.get_order(order_id)
        return {
            "order_id": order["order_id"],
            "status": order["status"],
            "estimated_delivery": order["estimated_delivery"],
            "tracking_number": order["tracking_number"],
        }

    def get_orders_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            repo = OrderRepository(db)
            return repo.get_by_customer(customer_id)

    # ── Support tickets ───────────────────────────────────────────────────────

    def create_support_ticket(
        self, customer_id: str, category: str, description: str
    ) -> dict[str, Any]:
        with SessionLocal() as db:
            repo = TicketRepository(db)
            return repo.create(customer_id, category, description)

    def get_customer_tickets(self, customer_id: str) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            repo = TicketRepository(db)
            return repo.get_by_customer(customer_id)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def schedule_callback(
        self,
        customer_id: str,
        preferred_time: str,
        reason: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """
        Schedule a callback.

        Idempotency is enforced by the database layer (CallbackRepository).
        """
        with SessionLocal() as db:
            repo = CallbackRepository(db)
            return repo.create(customer_id, preferred_time, reason, idempotency_key)
