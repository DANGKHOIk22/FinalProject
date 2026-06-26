"""
One-shot migration script: create the `user_threads` table in PostgreSQL
if it does not already exist.

Usage (from FinalProject/):
    .venv\Scripts\python.exe scripts\create_user_threads_table.py
"""
import sys
import os

# Ensure FinalProject/ is on sys.path so that `app.*` imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sqlalchemy import inspect
from app.database.db import engine, Base
from app.database.models import UserThread  # noqa: F401 – registers the model

TABLE_NAME = UserThread.__tablename__


def table_exists() -> bool:
    insp = inspect(engine)
    return insp.has_table(TABLE_NAME)


def main() -> None:
    print(f"🔍 Checking if '{TABLE_NAME}' table exists in PostgreSQL...")
    if table_exists():
        print(f"✅ Table '{TABLE_NAME}' already exists — nothing to do.")
        return

    print(f"📦 Creating table '{TABLE_NAME}'...")
    # Create ONLY the new table (without touching existing ones)
    Base.metadata.create_all(bind=engine, tables=[UserThread.__table__])
    print(f"✅ Table '{TABLE_NAME}' created successfully.")


if __name__ == "__main__":
    main()
