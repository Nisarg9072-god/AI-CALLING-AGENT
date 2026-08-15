"""
SQLAlchemy declarative models for the Company database.
"""

from __future__ import annotations

from sqlalchemy import Column, String, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from app.database import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    phone = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, nullable=False)
    pin_hash = Column(String, nullable=False)
    account_status = Column(String, nullable=False, default="active")

    orders = relationship("Order", back_populates="customer")
    tickets = relationship("Ticket", back_populates="customer")
    callbacks = relationship("Callback", back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False)
    status = Column(String, nullable=False)
    item = Column(String, nullable=False)
    tracking_number = Column(String, nullable=True)
    estimated_delivery = Column(String, nullable=True)
    delivery_address = Column(String, nullable=False)

    customer = relationship("Customer", back_populates="orders")


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False)
    category = Column(String, nullable=False)
    description = Column(String, nullable=False)
    status = Column(String, nullable=False, default="open")

    customer = relationship("Customer", back_populates="tickets")


class Callback(Base):
    __tablename__ = "callbacks"

    callback_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"), nullable=False)
    preferred_time = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    status = Column(String, nullable=False, default="scheduled")
    idempotency_key = Column(String, unique=True, index=True, nullable=False)

    customer = relationship("Customer", back_populates="callbacks")
