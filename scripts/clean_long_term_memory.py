"""
Clean (truncate) the user_long_term_memory table in Postgres.
Usage:
    uv run scripts/clean_long_term_memory.py          # truncate all
    uv run scripts/clean_long_term_memory.py --user_id <id>   # delete for a specific user
    uv run scripts/clean_long_term_memory.py --entity <name>  # delete for a specific entity
"""
import os
import sys
import argparse
import psycopg
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CleanLongTermMemory")


def clean(user_id: str = None, entity_name: str = None):
    db_url = os.getenv("POSTGRES_DATABASE_URL")
    if not db_url:
        logger.error("POSTGRES_DATABASE_URL not set.")
        return

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Show current row count
                cur.execute("SELECT COUNT(*) FROM user_long_term_memory")
                total = cur.fetchone()[0]
                logger.info(f"Current rows in user_long_term_memory: {total}")

                if user_id and entity_name:
                    cur.execute(
                        "DELETE FROM user_long_term_memory WHERE user_id = %s AND entity_name ILIKE %s",
                        (user_id, entity_name),
                    )
                elif user_id:
                    cur.execute("DELETE FROM user_long_term_memory WHERE user_id = %s", (user_id,))
                elif entity_name:
                    cur.execute("DELETE FROM user_long_term_memory WHERE entity_name ILIKE %s", (entity_name,))
                else:
                    cur.execute("TRUNCATE TABLE user_long_term_memory")

                deleted = cur.rowcount if cur.rowcount >= 0 else total
                conn.commit()

                cur.execute("SELECT COUNT(*) FROM user_long_term_memory")
                remaining = cur.fetchone()[0]

        logger.info(f"✅ Deleted {deleted} rows. Remaining: {remaining}")
    except Exception as e:
        logger.error(f"❌ Clean failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean user_long_term_memory table")
    parser.add_argument("--user_id", type=str, default=None, help="Delete rows for specific user_id")
    parser.add_argument("--entity", type=str, default=None, help="Delete rows for specific entity_name (case-insensitive)")
    args = parser.parse_args()
    clean(user_id=args.user_id, entity_name=args.entity)
