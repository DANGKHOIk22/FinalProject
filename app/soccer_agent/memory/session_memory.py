"""
Session Memory Manager

Manages conversation history token budget. When the formatted history string
exceeds SESSION_MEMORY_TOKEN_THRESHOLD, this module:
1. Loads cached SessionMemory from PostgreSQL (if available).
2. If no cache: triggers background LLM summarization → saves to PostgreSQL.
3. Always runs QueryUnderstandingPipeline in the hot path (1 LLM call only).

Token counting uses a fast character-based approximation (len // 4) since it is
only used for threshold decisions where precision is unnecessary.
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional

import psycopg
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.config.settings import settings

logger = logging.getLogger(__name__)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class UserContext(BaseModel):
    preferences: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)


class SessionMemory(BaseModel):
    conversation_state: str = ""
    user_context: UserContext = Field(default_factory=UserContext)
    shared_context: List[str] = Field(default_factory=list)
    open_discussion_threads: List[str] = Field(default_factory=list)
    scope: str = ""


# ── Manager ───────────────────────────────────────────────────────────────────

class SessionMemoryManager:
    """
    Token-aware conversation history manager.

    Hot path cost: 0 LLM calls (Path A) or 0 additional LLM calls beyond QU (Path B).
    Summarization is always async background → does not block the request.
    """

    def __init__(
        self,
        llm,
        token_threshold: int = 3000,
        recent_messages_to_keep: int = 5,
        cache_max_size: int = 256,
    ):
        self.llm = llm
        self.token_threshold = token_threshold
        self.recent_messages_to_keep = recent_messages_to_keep
        self._parser = PydanticOutputParser(pydantic_object=SessionMemory)
        self._cache: Dict[str, SessionMemory] = {}
        self._cache_max_size = cache_max_size

    # ── Token counting ────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Fast token estimate for threshold decisions. 1 token ~ 4 chars.
        Avoids a Gemini API round-trip (~100-300ms) since this is only used
        for the Path A/B threshold comparison, where precision is unnecessary.
        """
        return len(text) // 4

    # ── PostgreSQL persistence ────────────────────────────────────────────────

    def _db_load(self, session_id: str) -> Optional[SessionMemory]:
        """Synchronous DB read — wrapped via asyncio.to_thread in the async caller."""
        try:
            with psycopg.connect(settings.POSTGRES_DATABASE_URL) as conn:
                row = conn.execute(
                    "SELECT memory_json FROM session_memories WHERE session_id = %s",
                    (session_id,)
                ).fetchone()
            if row:
                return SessionMemory.model_validate_json(row[0])
        except Exception as e:
            logger.warning(f"[SessionMemory] load failed for {session_id}: {e}")
        return None

    def _db_save(self, session_id: str, memory: SessionMemory) -> None:
        """Synchronous DB upsert — wrapped via asyncio.to_thread in the async caller."""
        try:
            memory_json = memory.model_dump_json()
            with psycopg.connect(settings.POSTGRES_DATABASE_URL) as conn:
                conn.execute(
                    """
                    INSERT INTO session_memories (session_id, memory_json, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (session_id) DO UPDATE
                        SET memory_json = EXCLUDED.memory_json,
                            updated_at  = NOW()
                    """,
                    (session_id, memory_json)
                )
                conn.commit()
            logger.info(f"[SessionMemory] saved for session {session_id}")
        except Exception as e:
            logger.warning(f"[SessionMemory] save failed for {session_id}: {e}")

    async def load_cached_memory(self, session_id: str) -> Optional[SessionMemory]:
        """Load SessionMemory: check in-memory cache first, then PostgreSQL."""
        if session_id in self._cache:
            logger.debug(f"[SessionMemory] in-memory cache hit for {session_id}")
            return self._cache[session_id]
        memory = await asyncio.to_thread(self._db_load, session_id)
        if memory is not None:
            self._update_cache(session_id, memory)
        return memory

    async def save_session_memory(self, session_id: str, memory: SessionMemory) -> None:
        """Save SessionMemory to PostgreSQL and update in-memory cache."""
        self._update_cache(session_id, memory)
        await asyncio.to_thread(self._db_save, session_id, memory)

    def _update_cache(self, session_id: str, memory: SessionMemory) -> None:
        """Insert/update cache entry. Evicts oldest entry if over max size."""
        if session_id not in self._cache and len(self._cache) >= self._cache_max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[session_id] = memory

    # ── Summarization ─────────────────────────────────────────────────────────

    def _summarise_sync(self, messages: List[BaseMessage]) -> SessionMemory:
        """Call LLM synchronously to produce a SessionMemory from old messages."""
        from app.soccer_agent.prompts.session_memory import get_summarisation_prompt
        prompt_template = get_summarisation_prompt()
        conversation_block = self._messages_to_text(messages)
        prompt = prompt_template.invoke({
            "conversation_block": conversation_block,
            "format_instructions": self._parser.get_format_instructions(),
        })
        try:
            response = self.llm.invoke(prompt)
            response_text = response.text if hasattr(response, "text") else str(response)
            return self._parser.parse(response_text)
        except Exception as e:
            logger.warning(f"[SessionMemory] summarise failed: {e}. Using empty SessionMemory.")
            return SessionMemory()

    async def background_summarise(self, session_id: str, old_messages: List[BaseMessage]) -> None:
        """
        Run summarization in the background (called via asyncio.create_task).
        Summarizes old_messages → saves result to PostgreSQL.
        Does NOT block the current request.
        """
        try:
            logger.info(f"[SessionMemory] Background summarise started for {session_id} ({len(old_messages)} msgs)")
            memory = await asyncio.to_thread(self._summarise_sync, old_messages)
            await self.save_session_memory(session_id, memory)
            logger.info(f"[SessionMemory] Background summarise completed for {session_id}")
        except Exception as e:
            logger.error(f"[SessionMemory] Background summarise error for {session_id}: {e}")

    # ── History formatting ────────────────────────────────────────────────────

    def format_compressed_history(self, memory: SessionMemory, recent: List[BaseMessage]) -> str:
        """
        Format a SessionMemory + recent verbatim messages into the
        conversation_history string passed to planning and aggregator prompts.
        """
        parts = ["\n\n### TÓM TẮT LỊCH SỬ HỘI THOẠI (Session Memory):"]

        if memory.scope:
            parts.append(f"Phạm vi thảo luận: {memory.scope}")
        if memory.conversation_state:
            parts.append(f"Trạng thái hội thoại: {memory.conversation_state}")
        if memory.shared_context:
            parts.append("Thông tin đã xác lập:")
            for item in memory.shared_context:
                parts.append(f"  - {item}")
        if memory.user_context.preferences:
            parts.append("Sở thích người dùng:")
            for pref in memory.user_context.preferences:
                parts.append(f"  - {pref}")
        if memory.user_context.goals:
            parts.append("Mục tiêu người dùng:")
            for goal in memory.user_context.goals:
                parts.append(f"  - {goal}")
        if memory.open_discussion_threads:
            parts.append("Chủ đề đang thảo luận:")
            for thread in memory.open_discussion_threads:
                parts.append(f"  - {thread}")

        if recent:
            parts.append("\n### TIN NHẮN GẦN ĐÂY:")
            for msg in recent:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                parts.append(f"{role}: {msg.content}")

        return "\n".join(parts)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _messages_to_text(messages: List[BaseMessage]) -> str:
        lines = []
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
