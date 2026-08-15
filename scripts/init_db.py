"""
Initialize the database and populate with mock data.
Used for tests, evals, and local development.
"""

from __future__ import annotations

from app.database import Base, engine, SessionLocal
from app.company.models import Customer, Order

CUSTOMERS = [
    {
        "customer_id": "C001", "name": "Alice Johnson",
        "phone": "+15550101", "email": "alice@example.com",
        "pin_hash": "1234", "account_status": "active",
    },
    {
        "customer_id": "C002", "name": "Bob Martinez",
        "phone": "+15550202", "email": "bob@example.com",
        "pin_hash": "5678", "account_status": "active",
    },
    {
        "customer_id": "C003", "name": "Carol Williams",
        "phone": "+15550303", "email": "carol@example.com",
        "pin_hash": "9012", "account_status": "active",
    },
]

ORDERS = [
    {
        "order_id": "ORD-1001", "customer_id": "C001",
        "status": "shipped", "item": "Laptop Stand",
        "tracking_number": "TRK-ALPHA-7823",
        "estimated_delivery": "2026-08-17",
        "delivery_address": "123 Main St, Springfield",
    },
    {
        "order_id": "ORD-1002", "customer_id": "C001",
        "status": "delivered", "item": "USB Hub",
        "tracking_number": "TRK-BETA-4491",
        "estimated_delivery": "2026-08-10",
        "delivery_address": "123 Main St, Springfield",
    },
    {
        "order_id": "ORD-2001", "customer_id": "C002",
        "status": "delayed", "item": "Mechanical Keyboard",
        "tracking_number": "TRK-GAMMA-3310",
        "estimated_delivery": "2026-08-20",
        "delivery_address": "456 Oak Ave, Shelbyville",
    },
    {
        "order_id": "ORD-3001", "customer_id": "C003",
        "status": "processing", "item": "Monitor",
        "tracking_number": None,
        "estimated_delivery": "2026-08-22",
        "delivery_address": "789 Pine Rd, Capital City",
    },
]

def init_db():
    print("Creating database tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    print("Populating mock data...")
    with SessionLocal() as db:
        for c_data in CUSTOMERS:
            db.add(Customer(**c_data))
        for o_data in ORDERS:
            db.add(Order(**o_data))
        db.commit()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
