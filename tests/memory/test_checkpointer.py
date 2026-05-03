"""Unit tests for the checkpointer lifecycle."""
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub out langgraph.checkpoint.postgres before the module is imported
_pg_stub = ModuleType("langgraph.checkpoint.postgres")
_pg_aio_stub = ModuleType("langgraph.checkpoint.postgres.aio")
_pg_aio_stub.AsyncPostgresSaver = MagicMock
sys.modules.setdefault("langgraph.checkpoint.postgres", _pg_stub)
sys.modules.setdefault("langgraph.checkpoint.postgres.aio", _pg_aio_stub)

import app.soccer_agent.memory.checkpointer as cp_module  # noqa: E402


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons before each test."""
    cp_module._pool = None
    cp_module._checkpointer = None
    yield
    cp_module._pool = None
    cp_module._checkpointer = None


class TestGetCheckpointer:
    def test_raises_if_not_initialized(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            cp_module.get_checkpointer()

    def test_returns_checkpointer_after_init(self):
        sentinel = MagicMock()
        cp_module._checkpointer = sentinel
        assert cp_module.get_checkpointer() is sentinel


class TestCloseCheckpointer:
    @pytest.mark.asyncio
    async def test_close_when_pool_is_none_is_safe(self):
        await cp_module.close_checkpointer()  # should not raise

    @pytest.mark.asyncio
    async def test_close_resets_singletons(self):
        mock_pool = AsyncMock()
        cp_module._pool = mock_pool
        cp_module._checkpointer = MagicMock()

        await cp_module.close_checkpointer()

        mock_pool.close.assert_awaited_once()
        assert cp_module._pool is None
        assert cp_module._checkpointer is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        await cp_module.close_checkpointer()
        await cp_module.close_checkpointer()  # second call must not raise


class TestInitCheckpointer:
    @pytest.mark.asyncio
    async def test_init_sets_singletons(self):
        mock_pool = AsyncMock()
        mock_saver = AsyncMock()
        mock_saver.setup = AsyncMock()

        with patch("app.soccer_agent.memory.checkpointer.AsyncConnectionPool", return_value=mock_pool), \
             patch("app.soccer_agent.memory.checkpointer.AsyncPostgresSaver", return_value=mock_saver):
            result = await cp_module.init_checkpointer()

        assert result is mock_saver
        assert cp_module._checkpointer is mock_saver
        assert cp_module._pool is mock_pool
        mock_pool.open.assert_awaited_once()
        mock_saver.setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_init_pool_constructed_with_correct_kwargs(self):
        mock_pool = AsyncMock()
        mock_saver = AsyncMock()
        mock_saver.setup = AsyncMock()

        with patch("app.soccer_agent.memory.checkpointer.AsyncConnectionPool", return_value=mock_pool) as mock_cls, \
             patch("app.soccer_agent.memory.checkpointer.AsyncPostgresSaver", return_value=mock_saver):
            await cp_module.init_checkpointer()

        _, kwargs = mock_cls.call_args
        assert kwargs["max_size"] == 20
        assert kwargs["kwargs"]["autocommit"] is True
        assert kwargs["kwargs"]["prepare_threshold"] is None
        assert kwargs["open"] is False
