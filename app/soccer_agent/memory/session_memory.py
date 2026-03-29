"""
Session Memory Manager

Manages conversation history token budget. When the formatted history string
exceeds SESSION_MEMORY_TOKEN_THRESHOLD, this module:
1. Loads cached SessionMemory from Redis (if available).
2. If no cache: triggers background LLM summarization → saves to Redis.
3. After every turn that has tool calls: triggers background incremental update
   of the cached SessionMemory with new tool findings.

Token counting uses a fast character-based approximation (len // 4) since it is
only used for threshold decisions where precision is unnecessary.

Redis key: soccer_agent:session_memory:{session_id}
Redis TTL: SESSION_MEMORY_REDIS_TTL (default 24h)
"""
import asyncio
import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.config.config import SESSION_MEMORY_REDIS_TTL

logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = "soccer_agent:session_memory:"


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class UserContext(BaseModel):
    preferences: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)


class ToolFinding(BaseModel):
    """Kết quả chi tiết từ một tool call — được dùng làm long-term memory."""
    tool_name: str = Field(description="Tên tool đã gọi")
    input_summary: str = Field(description="Tóm tắt ngắn input của tool (query hoặc đối số chính)")
    key_facts: List[str] = Field(
        default_factory=list,
        description=(
            "Các thông tin quan trọng từ response và artifact của tool. "
            "Giữ đủ chi tiết để tái sử dụng mà không cần gọi lại tool. "
            "Mỗi fact là một câu ngắn gọn, đầy đủ thông tin."
        )
    )


class SessionMemory(BaseModel):
    conversation_state: str = Field(
        default="",
        description="Tóm tắt tiến trình hội thoại — đang ở giai đoạn nào, đã giải quyết được gì"
    )
    user_context: UserContext = Field(default_factory=UserContext)
    confirmed_entities: List[str] = Field(
        default_factory=list,
        description=(
            "Thực thể đã xác nhận dạng 'Tên (loại) — fact chính'. "
            "Dùng cho pronoun resolution trong Query Understanding. "
            "Ví dụ: 'Chelsea FC (đội bóng) — EPL 2014-15, vô địch với 87 điểm'"
        )
    )
    tool_findings: List[ToolFinding] = Field(
        default_factory=list,
        description="Kết quả chi tiết từ tất cả tool calls qua các turn trong session"
    )
    open_discussion_threads: List[str] = Field(default_factory=list)
    scope: str = Field(default="")


# ── Manager ───────────────────────────────────────────────────────────────────

class SessionMemoryManager:
    """
    Token-aware conversation history manager backed by Redis.

    Hot path: 0 LLM calls (Path A) or at most the QU call (Path B).
    All summarization and incremental updates are async background tasks.

    Redis key: soccer_agent:session_memory:{session_id}
    """

    def __init__(
        self,
        llm,
        token_threshold: int = 3000,
        recent_messages_to_keep: int = 5,
    ):
        self.llm = llm
        self.token_threshold = token_threshold
        self.recent_messages_to_keep = recent_messages_to_keep
        self._parser = PydanticOutputParser(pydantic_object=SessionMemory)

        # Lazy import to avoid circular deps at module load time
        self._redis = None

    @property
    def redis(self):
        if self._redis is None:
            from app.cache.standard_cache import standard_cache
            self._redis = standard_cache.client
        return self._redis

    def _redis_key(self, session_id: str) -> str:
        return f"{REDIS_KEY_PREFIX}{session_id}"

    # ── Token counting ────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Fast token estimate for threshold decisions. 1 token ~ 4 chars."""
        return len(text) // 4

    # ── Redis persistence ─────────────────────────────────────────────────────

    def _redis_load(self, session_id: str) -> Optional[SessionMemory]:
        """Synchronous Redis read. Wrapped via asyncio.to_thread by callers."""
        try:
            raw = self.redis.get(self._redis_key(session_id))
            if raw:
                return SessionMemory.model_validate_json(raw)
        except Exception as e:
            logger.warning(f"[SessionMemory] Redis load failed for {session_id}: {e}")
        return None

    def _redis_save(self, session_id: str, memory: SessionMemory) -> None:
        """Synchronous Redis upsert with TTL. Wrapped via asyncio.to_thread by callers."""
        try:
            self.redis.setex(
                self._redis_key(session_id),
                SESSION_MEMORY_REDIS_TTL,
                memory.model_dump_json(),
            )
            logger.info(f"[SessionMemory] Redis saved for session {session_id}")
        except Exception as e:
            logger.warning(f"[SessionMemory] Redis save failed for {session_id}: {e}")

    def _redis_exists(self, session_id: str) -> bool:
        """Check if a SessionMemory key exists in Redis."""
        try:
            return bool(self.redis.exists(self._redis_key(session_id)))
        except Exception as e:
            logger.warning(f"[SessionMemory] Redis exists check failed for {session_id}: {e}")
            return False

    async def load_cached_memory(self, session_id: str) -> Optional[SessionMemory]:
        """Load SessionMemory from Redis (non-blocking via thread)."""
        return await asyncio.to_thread(self._redis_load, session_id)

    async def save_session_memory(self, session_id: str, memory: SessionMemory) -> None:
        """Save SessionMemory to Redis (non-blocking via thread)."""
        await asyncio.to_thread(self._redis_save, session_id, memory)

    async def has_cached_memory(self, session_id: str) -> bool:
        """Check if a SessionMemory exists for this session (fast key existence check)."""
        return await asyncio.to_thread(self._redis_exists, session_id)

    # ── Full summarization (first-time, background) ───────────────────────────

    def _summarise_sync(self, messages: List[BaseMessage]) -> SessionMemory:
        """Call LLM to produce a SessionMemory from the old portion of history."""
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
        First-time summarization. Called via asyncio.create_task — does NOT block request.
        Summarizes old_messages → saves to Redis.
        """
        try:
            logger.info(f"[SessionMemory] Background summarise started for {session_id} ({len(old_messages)} msgs)")
            memory = await asyncio.to_thread(self._summarise_sync, old_messages)
            await self.save_session_memory(session_id, memory)
            logger.info(f"[SessionMemory] Background summarise completed for {session_id}")
        except Exception as e:
            logger.error(f"[SessionMemory] Background summarise error for {session_id}: {e}")

    # ── Incremental update (after each turn with tool calls) ──────────────────

    def _update_sync(self, session_id: str, turn_tool_summary: str) -> None:
        """
        Load existing SessionMemory from Redis, merge in new tool findings
        from the current turn, then save back.
        """
        from app.soccer_agent.prompts.session_memory import get_update_prompt

        existing = self._redis_load(session_id)
        if existing is None:
            logger.warning(f"[SessionMemory] Update skipped — no existing memory for {session_id}")
            return

        update_parser = PydanticOutputParser(pydantic_object=SessionMemory)
        prompt_template = get_update_prompt()
        prompt = prompt_template.invoke({
            "existing_memory": existing.model_dump_json(indent=2),
            "turn_tool_summary": turn_tool_summary,
            "format_instructions": update_parser.get_format_instructions(),
        })
        try:
            response = self.llm.invoke(prompt)
            response_text = response.text if hasattr(response, "text") else str(response)
            updated = update_parser.parse(response_text)
            self._redis_save(session_id, updated)
            logger.info(
                f"[SessionMemory] Incremental update done for {session_id}. "
                f"tool_findings: {len(existing.tool_findings)} → {len(updated.tool_findings)}"
            )
        except Exception as e:
            logger.warning(f"[SessionMemory] Incremental update failed for {session_id}: {e}")

    async def background_update(self, session_id: str, turn_tool_summary: str) -> None:
        """
        Incremental update after a turn that has tool calls.
        Called via asyncio.create_task — does NOT block request.
        """
        try:
            await asyncio.to_thread(self._update_sync, session_id, turn_tool_summary)
        except Exception as e:
            logger.error(f"[SessionMemory] Background update error for {session_id}: {e}")

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

        if memory.confirmed_entities:
            parts.append("Thực thể đã xác nhận:")
            for entity in memory.confirmed_entities:
                parts.append(f"  - {entity}")

        if memory.tool_findings:
            parts.append("Kết quả từ các tool đã gọi:")
            for finding in memory.tool_findings:
                parts.append(f"  [{finding.tool_name}] Input: {finding.input_summary}")
                for fact in finding.key_facts:
                    parts.append(f"    • {fact}")

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
