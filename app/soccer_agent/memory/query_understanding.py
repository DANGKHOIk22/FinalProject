"""
Query Understanding Pipeline

Analyzes the user's query in the context of session memory and recent messages.
Produces a clarified_query with all pronouns/references resolved to specific
soccer entities.

Always runs in the hot path (Path B) with a single LLM call.
Uses execution_llm (Flash Lite) with no thinking budget for speed.
"""
import logging
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.soccer_agent.memory.session_memory import SessionMemory

logger = logging.getLogger(__name__)


class QueryUnderstandingOutput(BaseModel):
    clarified_query: str = Field(description="User query with all pronouns and references resolved to specific soccer entities")
    is_ambiguous: bool = Field(default=False, description="True if the query cannot be resolved unambiguously")
    clarifying_questions: List[str] = Field(default_factory=list, description="Questions to ask the user when is_ambiguous=True")


class QueryUnderstandingPipeline:
    """
    Three-step ambiguity detection pipeline adapted for soccer domain.
    Steps: Topic Link → Referent Count → Sufficiency.

    Always returns clarified_query (best-guess even when ambiguous).
    Fallback: returns original user_query on any parse error.
    """

    def __init__(self, llm):
        self.llm = llm
        self._parser = PydanticOutputParser(pydantic_object=QueryUnderstandingOutput)

    async def run(
        self,
        user_query: str,
        session_memory: SessionMemory,
        recent_messages: List[BaseMessage],
    ) -> QueryUnderstandingOutput:
        from app.soccer_agent.prompts.query_understanding import get_query_understanding_prompt

        prompt_template = get_query_understanding_prompt()
        memory_text = self._memory_to_text(session_memory)
        recent_text = self._messages_to_text(recent_messages)

        prompt = prompt_template.invoke({
            "user_query": user_query,
            "session_memory": memory_text,
            "recent_messages": recent_text,
            "format_instructions": self._parser.get_format_instructions(),
        })

        try:
            response = await self.llm.ainvoke(prompt)
            response_text = response.text if hasattr(response, "text") else str(response)
            result = self._parser.parse(response_text)
            logger.info(
                f"[QU] clarified_query='{result.clarified_query}' "
                f"is_ambiguous={result.is_ambiguous}"
            )
            return result
        except Exception as e:
            logger.warning(f"[QU] Parse failed: {e}. Returning original query.")
            return QueryUnderstandingOutput(clarified_query=user_query)

    @staticmethod
    def _memory_to_text(memory: SessionMemory) -> str:
        if not any([
            memory.scope,
            memory.conversation_state,
            memory.confirmed_entities,
            memory.open_discussion_threads,
            memory.tool_findings,
        ]):
            return "Không có bộ nhớ phiên."

        parts = []
        if memory.scope:
            parts.append(f"Phạm vi: {memory.scope}")
        if memory.conversation_state:
            parts.append(f"Trạng thái: {memory.conversation_state}")
        if memory.confirmed_entities:
            parts.append("Thực thể đã xác nhận: " + "; ".join(memory.confirmed_entities))
        if memory.open_discussion_threads:
            parts.append("Chủ đề đang mở: " + "; ".join(memory.open_discussion_threads))
        if memory.tool_findings:
            parts.append("Kết quả tìm kiếm gần đây:")
            for f in memory.tool_findings[-5:]:
                facts_preview = "; ".join(f.key_facts[:2])
                parts.append(f"  [{f.tool_name}] {f.input_summary}: {facts_preview}")
        return "\n".join(parts)

    @staticmethod
    def _messages_to_text(messages: List[BaseMessage]) -> str:
        if not messages:
            return "Không có tin nhắn gần đây."
        lines = []
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
