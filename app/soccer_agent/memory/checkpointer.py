"""Async PostgreSQL checkpointer singleton for LangGraph short-term memory."""
import logging

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config.settings import settings

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None


async def init_checkpointer() -> AsyncPostgresSaver:
    """Initialize the async connection pool and LangGraph checkpointer.

    Call once during app startup (lifespan). The checkpointer persists
    AgentState (including messages) per thread_id to PostgreSQL automatically.
    """
    global _pool, _checkpointer
    _pool = AsyncConnectionPool(
        conninfo=settings.POSTGRES_DATABASE_URL,
        max_size=20,
        kwargs={"prepare_threshold": None, "autocommit": True},
        open=False,
    )
    await _pool.open()
    _checkpointer = AsyncPostgresSaver(_pool)
    await _checkpointer.setup()
    logger.info("✅ LangGraph checkpointer initialized")
    return _checkpointer


async def close_checkpointer() -> None:
    """Gracefully close pool on app shutdown."""
    global _pool, _checkpointer
    if _pool:
        await _pool.close()
    _pool = None
    _checkpointer = None
    logger.info("LangGraph checkpointer closed")


def get_checkpointer() -> AsyncPostgresSaver:
    """Return the initialized checkpointer (raises if not yet initialized)."""
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized. Call init_checkpointer() in lifespan.")
    return _checkpointer
