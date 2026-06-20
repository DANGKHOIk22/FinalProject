"""
Migration script: Create `user_threads` table if it does not exist.

Run:
    python scripts/database/create_user_threads_table.py

This script is idempotent — safe to run multiple times.
"""
import sys
import os

# Ensure FinalProject/ is on sys.path so app.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import inspect, text
from app.database.db import engine, Base


def table_exists(table_name: str) -> bool:
    """Check if a table already exists in the database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def create_table():
    """Create the user_threads table if it does not already exist."""
    if table_exists("user_threads"):
        print("✅ Table `user_threads` already exists — nothing to do.")
        return

    # Import models to register them with Base.metadata
    import app.database.models  # noqa: F401 — registers UserThread with Base

    Base.metadata.create_all(bind=engine, tables=[app.database.models.UserThread.__table__])
    print("✅ Table `user_threads` created successfully.")


if __name__ == "__main__":
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ PostgreSQL connection verified.")
    except Exception as e:
        print(f"❌ Cannot connect to PostgreSQL: {e}")
        sys.exit(1)

    create_table()
