"""
Database configuration and session management.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# Since the Agent loop is synchronous, we use the synchronous SQLite driver.
# If the config specifies aiosqlite, we replace it with the standard sqlite driver.
db_url = settings.database_url.replace("sqlite+aiosqlite", "sqlite")

# Ensure the data directory exists
if db_url.startswith("sqlite:///"):
    db_path = db_url.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

engine = create_engine(
    db_url,
    connect_args={"check_same_thread": False}, # Needed for SQLite with multiple threads
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency for getting a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
