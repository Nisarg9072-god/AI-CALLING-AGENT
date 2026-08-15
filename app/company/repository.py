"""
Repositories for database access.
Isolates SQLAlchemy logic from the business service.
"""

from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session

from app.company.models import Customer, Order, Ticket, Callback


class CustomerRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, customer_id: str) -> dict[str, Any] | None:
        customer = self.db.query(Customer).filter(Customer.customer_id == customer_id).first()
        if not customer:
            return None
        return {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "phone": customer.phone,
            "email": customer.email,
            "pin_hash": customer.pin_hash,
            "account_status": customer.account_status,
        }

    def get_by_phone(self, phone: str) -> dict[str, Any] | None:
        customer = self.db.query(Customer).filter(Customer.phone == phone).first()
        if not customer:
            return None
        return {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "phone": customer.phone,
            "email": customer.email,
            "pin_hash": customer.pin_hash,
            "account_status": customer.account_status,
        }


class OrderRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, order_id: str) -> dict[str, Any] | None:
        order = self.db.query(Order).filter(Order.order_id == order_id).first()
        if not order:
            return None
        return {
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "status": order.status,
            "item": order.item,
            "tracking_number": order.tracking_number,
            "estimated_delivery": order.estimated_delivery,
            "delivery_address": order.delivery_address,
        }

    def get_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        orders = self.db.query(Order).filter(Order.customer_id == customer_id).all()
        return [
            {
                "order_id": order.order_id,
                "customer_id": order.customer_id,
                "status": order.status,
                "item": order.item,
                "tracking_number": order.tracking_number,
                "estimated_delivery": order.estimated_delivery,
                "delivery_address": order.delivery_address,
            }
            for order in orders
        ]


class TicketRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, customer_id: str, category: str, description: str) -> dict[str, Any]:
        import uuid
        ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
        ticket = Ticket(
            ticket_id=ticket_id,
            customer_id=customer_id,
            category=category,
            description=description,
            status="open",
        )
        self.db.add(ticket)
        self.db.commit()
        self.db.refresh(ticket)
        return {
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "category": ticket.category,
            "description": ticket.description,
            "status": ticket.status,
        }

    def get_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        tickets = self.db.query(Ticket).filter(Ticket.customer_id == customer_id).all()
        return [
            {
                "ticket_id": ticket.ticket_id,
                "customer_id": ticket.customer_id,
                "category": ticket.category,
                "description": ticket.description,
                "status": ticket.status,
            }
            for ticket in tickets
        ]


class CallbackRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, customer_id: str, preferred_time: str, reason: str, idempotency_key: str) -> dict[str, Any]:
        # Check idempotency first
        existing = self.db.query(Callback).filter(Callback.idempotency_key == idempotency_key).first()
        if existing:
            return {
                "callback_id": existing.callback_id,
                "customer_id": existing.customer_id,
                "preferred_time": existing.preferred_time,
                "reason": existing.reason,
                "status": existing.status,
                "idempotency_key": existing.idempotency_key,
                "_idempotent": True,
            }
        
        import uuid
        callback_id = f"CB-{uuid.uuid4().hex[:6].upper()}"
        callback = Callback(
            callback_id=callback_id,
            customer_id=customer_id,
            preferred_time=preferred_time,
            reason=reason,
            status="scheduled",
            idempotency_key=idempotency_key,
        )
        self.db.add(callback)
        self.db.commit()
        self.db.refresh(callback)
        return {
            "callback_id": callback.callback_id,
            "customer_id": callback.customer_id,
            "preferred_time": callback.preferred_time,
            "reason": callback.reason,
            "status": callback.status,
            "idempotency_key": callback.idempotency_key,
        }
