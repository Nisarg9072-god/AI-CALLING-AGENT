"""
In-memory company data store — Phase 3 stub.

IMPORTANT: This is a temporary implementation for Phases 3–4.
Phase 4 replaces this with real SQLite + SQLAlchemy repositories.

The Agent must NEVER import from this file directly.
Access path: Agent → Tool → CompanyService → Repository → Database
"""

from __future__ import annotations

from typing import Any


# ── Mock data ──────────────────────────────────────────────────────────────────

CUSTOMERS: dict[str, dict[str, Any]] = {
    "C001": {
        "customer_id": "C001", "name": "Alice Johnson",
        "phone": "+15550101", "email": "alice@example.com",
        "pin_hash": "1234",   # plain for now; Phase 4 will hash properly
        "account_status": "active",
    },
    "C002": {
        "customer_id": "C002", "name": "Bob Martinez",
        "phone": "+15550202", "email": "bob@example.com",
        "pin_hash": "5678", "account_status": "active",
    },
    "C003": {
        "customer_id": "C003", "name": "Carol Williams",
        "phone": "+15550303", "email": "carol@example.com",
        "pin_hash": "9012", "account_status": "active",
    },
}

ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {
        "order_id": "ORD-1001", "customer_id": "C001",
        "status": "shipped", "item": "Laptop Stand",
        "tracking_number": "TRK-ALPHA-7823",
        "estimated_delivery": "2026-08-17",
        "delivery_address": "123 Main St, Springfield",
    },
    "ORD-1002": {
        "order_id": "ORD-1002", "customer_id": "C001",
        "status": "delivered", "item": "USB Hub",
        "tracking_number": "TRK-BETA-4491",
        "estimated_delivery": "2026-08-10",
        "delivery_address": "123 Main St, Springfield",
    },
    "ORD-2001": {
        "order_id": "ORD-2001", "customer_id": "C002",
        "status": "delayed", "item": "Mechanical Keyboard",
        "tracking_number": "TRK-GAMMA-3310",
        "estimated_delivery": "2026-08-20",
        "delivery_address": "456 Oak Ave, Shelbyville",
    },
    "ORD-3001": {
        "order_id": "ORD-3001", "customer_id": "C003",
        "status": "processing", "item": "Monitor",
        "tracking_number": None,
        "estimated_delivery": "2026-08-22",
        "delivery_address": "789 Pine Rd, Capital City",
    },
}

# Runtime stores (mutable during a process run)
_tickets: dict[str, dict[str, Any]] = {}
_callbacks: dict[str, dict[str, Any]] = {}
_ticket_counter = 1000
_callback_counter = 100


class CompanyServiceError(Exception):
    """Raised when a business rule is violated or data is not found."""


class CompanyService:
    """
    Business logic layer — the ONLY gateway to company data.

    The Agent → Tool → CompanyService path is enforced by architecture.
    Tools call CompanyService methods; CompanyService accesses data stores.
    No tool reads CUSTOMERS/ORDERS directly.
    """

    # ── Customer ───────────────────────────────────────────────────────────────

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Return customer data (never includes PIN)."""
        customer = CUSTOMERS.get(customer_id)
        if not customer:
            raise CompanyServiceError(f"Customer '{customer_id}' not found.")
        # Never return the PIN hash
        return {k: v for k, v in customer.items() if k != "pin_hash"}

    def verify_customer(self, customer_id: str, pin: str) -> bool:
        """Verify customer identity. Returns True if PIN matches."""
        customer = CUSTOMERS.get(customer_id)
        if not customer:
            raise CompanyServiceError(f"Customer '{customer_id}' not found.")
        return customer.get("pin_hash") == pin

    def find_customer_by_phone(self, phone: str) -> dict[str, Any] | None:
        """Look up a customer by phone number."""
        for c in CUSTOMERS.values():
            if c["phone"] == phone:
                return {k: v for k, v in c.items() if k != "pin_hash"}
        return None

    # ── Orders ─────────────────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> dict[str, Any]:
        order = ORDERS.get(order_id)
        if not order:
            raise CompanyServiceError(f"Order '{order_id}' not found.")
        return dict(order)

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
        return [dict(o) for o in ORDERS.values() if o["customer_id"] == customer_id]

    # ── Support tickets ────────────────────────────────────────────────────────

    def create_support_ticket(
        self, customer_id: str, category: str, description: str
    ) -> dict[str, Any]:
        global _ticket_counter
        _ticket_counter += 1
        ticket_id = f"TKT-{_ticket_counter}"
        ticket = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "category": category,
            "description": description,
            "status": "open",
        }
        _tickets[ticket_id] = ticket
        return dict(ticket)

    def get_customer_tickets(self, customer_id: str) -> list[dict[str, Any]]:
        return [t for t in _tickets.values() if t["customer_id"] == customer_id]

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def schedule_callback(
        self,
        customer_id: str,
        preferred_time: str,
        reason: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """
        Schedule a callback.

        Idempotency: if the same idempotency_key has already been used,
        return the existing callback without creating a new one.
        """
        # Check idempotency at the service layer too (belt-and-suspenders)
        for cb in _callbacks.values():
            if cb.get("idempotency_key") == idempotency_key:
                return {**cb, "_idempotent": True}

        global _callback_counter
        _callback_counter += 1
        callback_id = f"CB-{_callback_counter}"
        callback = {
            "callback_id": callback_id,
            "customer_id": customer_id,
            "preferred_time": preferred_time,
            "reason": reason,
            "status": "scheduled",
            "idempotency_key": idempotency_key,
        }
        _callbacks[callback_id] = callback
        return dict(callback)
