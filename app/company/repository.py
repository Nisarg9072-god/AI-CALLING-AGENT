"""
Company in-memory repository — seeded with realistic sample data.

Uses plain Python dicts as the backing store (no SQLite needed for this demo).
All IDs are human-readable strings so the LLM can easily reference them.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Any


# ── Data models (simple dataclasses — no Pydantic overhead needed) ─────────────


@dataclass
class Customer:
    customer_id: str
    name: str
    phone: str
    email: str
    pin: str  # 4-digit verification PIN
    account_status: str = "active"  # active | suspended


@dataclass
class Order:
    order_id: str
    customer_id: str
    status: str  # pending | processing | shipped | delivered | cancelled | delayed
    items: list[dict[str, Any]] = field(default_factory=list)
    total_amount: float = 0.0
    delivery_address: str = ""
    estimated_delivery: str = ""
    tracking_number: str = ""
    placed_at: str = ""


@dataclass
class SupportTicket:
    ticket_id: str
    customer_id: str
    issue_type: str  # billing | technical | delivery | general
    description: str
    status: str = "open"  # open | in_progress | resolved | closed
    created_at: str = ""


@dataclass
class Callback:
    callback_id: str
    customer_id: str
    scheduled_time: str
    reason: str
    status: str = "scheduled"  # scheduled | completed | cancelled


# ── Repository ─────────────────────────────────────────────────────────────────


class CompanyRepository:
    """
    In-memory data store for the mock company.

    All mutations return new state — no references leak out. The service layer
    is the only thing that calls this repository.
    """

    def __init__(self) -> None:
        self._customers: dict[str, Customer] = {}
        self._orders: dict[str, Order] = {}
        self._tickets: dict[str, SupportTicket] = {}
        self._callbacks: dict[str, Callback] = {}
        self._seed()

    # ── Seeding ────────────────────────────────────────────────────────────────

    def _seed(self) -> None:
        """Populate with realistic demo data."""
        now = datetime.now(timezone.utc)
        fmt = "%Y-%m-%d"

        # Customers
        customers = [
            Customer("C001", "Alice Johnson",    "+1-555-0101", "alice@example.com",    "1234"),
            Customer("C002", "Bob Martinez",     "+1-555-0202", "bob@example.com",      "5678"),
            Customer("C003", "Carol Williams",   "+1-555-0303", "carol@example.com",    "9012"),
            Customer("C004", "David Chen",       "+1-555-0404", "david@example.com",    "3456"),
            Customer("C005", "Emma Thompson",    "+1-555-0505", "emma@example.com",     "7890",
                     account_status="suspended"),
        ]
        for c in customers:
            self._customers[c.customer_id] = c

        # Orders
        orders = [
            Order(
                order_id="ORD-1001", customer_id="C001", status="shipped",
                items=[{"name": "Laptop Stand", "qty": 1, "price": 49.99},
                       {"name": "USB-C Hub",    "qty": 2, "price": 29.99}],
                total_amount=109.97,
                delivery_address="123 Oak Street, Austin, TX 78701",
                estimated_delivery=(now + timedelta(days=2)).strftime(fmt),
                tracking_number="TRK-ALPHA-7823",
                placed_at=(now - timedelta(days=3)).strftime(fmt),
            ),
            Order(
                order_id="ORD-1002", customer_id="C001", status="delivered",
                items=[{"name": "Mechanical Keyboard", "qty": 1, "price": 149.99}],
                total_amount=149.99,
                delivery_address="123 Oak Street, Austin, TX 78701",
                estimated_delivery=(now - timedelta(days=5)).strftime(fmt),
                tracking_number="TRK-BETA-3341",
                placed_at=(now - timedelta(days=10)).strftime(fmt),
            ),
            Order(
                order_id="ORD-2001", customer_id="C002", status="delayed",
                items=[{"name": "Monitor 27\"", "qty": 1, "price": 399.99}],
                total_amount=399.99,
                delivery_address="456 Pine Ave, Seattle, WA 98101",
                estimated_delivery=(now + timedelta(days=7)).strftime(fmt),
                tracking_number="TRK-GAMMA-9917",
                placed_at=(now - timedelta(days=5)).strftime(fmt),
            ),
            Order(
                order_id="ORD-3001", customer_id="C003", status="processing",
                items=[{"name": "Webcam HD", "qty": 1, "price": 89.99}],
                total_amount=89.99,
                delivery_address="789 Elm Road, Chicago, IL 60601",
                estimated_delivery=(now + timedelta(days=4)).strftime(fmt),
                tracking_number="",
                placed_at=(now - timedelta(days=1)).strftime(fmt),
            ),
            Order(
                order_id="ORD-4001", customer_id="C004", status="cancelled",
                items=[{"name": "Headphones", "qty": 1, "price": 199.99}],
                total_amount=199.99,
                delivery_address="321 Maple Dr, New York, NY 10001",
                estimated_delivery="",
                tracking_number="",
                placed_at=(now - timedelta(days=2)).strftime(fmt),
            ),
        ]
        for o in orders:
            self._orders[o.order_id] = o

        # Support tickets
        tickets = [
            SupportTicket(
                ticket_id="TKT-5001", customer_id="C002",
                issue_type="delivery",
                description="Order ORD-2001 is delayed — customer asking for ETA",
                status="open",
                created_at=(now - timedelta(days=1)).strftime(fmt),
            ),
            SupportTicket(
                ticket_id="TKT-5002", customer_id="C001",
                issue_type="billing",
                description="Charged twice for ORD-1002",
                status="resolved",
                created_at=(now - timedelta(days=8)).strftime(fmt),
            ),
        ]
        for t in tickets:
            self._tickets[t.ticket_id] = t

    # ── Customer CRUD ──────────────────────────────────────────────────────────

    def get_customer(self, customer_id: str) -> Customer | None:
        return deepcopy(self._customers.get(customer_id))

    def find_customer_by_phone(self, phone: str) -> Customer | None:
        for c in self._customers.values():
            if c.phone == phone:
                return deepcopy(c)
        return None

    def list_customers(self) -> list[Customer]:
        return [deepcopy(c) for c in self._customers.values()]

    # ── Order CRUD ─────────────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> Order | None:
        return deepcopy(self._orders.get(order_id))

    def get_orders_for_customer(self, customer_id: str) -> list[Order]:
        return [deepcopy(o) for o in self._orders.values() if o.customer_id == customer_id]

    def update_order_address(self, order_id: str, new_address: str) -> bool:
        if order_id not in self._orders:
            return False
        self._orders[order_id].delivery_address = new_address
        return True

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self._orders:
            return False
        if self._orders[order_id].status in ("delivered", "cancelled"):
            return False
        self._orders[order_id].status = "cancelled"
        return True

    # ── Support ticket CRUD ────────────────────────────────────────────────────

    def get_ticket(self, ticket_id: str) -> SupportTicket | None:
        return deepcopy(self._tickets.get(ticket_id))

    def get_tickets_for_customer(self, customer_id: str) -> list[SupportTicket]:
        return [deepcopy(t) for t in self._tickets.values() if t.customer_id == customer_id]

    def create_ticket(self, ticket: SupportTicket) -> SupportTicket:
        self._tickets[ticket.ticket_id] = ticket
        return deepcopy(ticket)

    # ── Callback CRUD ──────────────────────────────────────────────────────────

    def get_callbacks_for_customer(self, customer_id: str) -> list[Callback]:
        return [deepcopy(c) for c in self._callbacks.values() if c.customer_id == customer_id]

    def create_callback(self, callback: Callback) -> Callback:
        self._callbacks[callback.callback_id] = callback
        return deepcopy(callback)


# ── Module-level singleton ─────────────────────────────────────────────────────

_repository: CompanyRepository | None = None


def get_repository() -> CompanyRepository:
    global _repository
    if _repository is None:
        _repository = CompanyRepository()
    return _repository
