import logging
import uuid
import asyncio
from langfuse import get_client
from langchain_core.messages import AIMessage
from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph.state import RunnableConfig
from langgraph.types import Command
from langgraph.graph import END

from app.schema.soccer_agent.state import AgentState
from app.soccer_agent.memory.query_understanding import QueryUnderstandingPipeline

logger = logging.getLogger(__name__)

class UnderstandUserMessageNode:
    def __init__(self, query_understanding: QueryUnderstandingPipeline):
        self.query_understanding = query_understanding

    async def understand_user_message(self, state: AgentState, config: RunnableConfig):
        """
        This node runs QueryUnderstanding to resolve pronouns/abbreviations and handle domain jargon.
        If the query is ambiguous, it will ask for clarification immediately.
        """
        messages = state.get("messages", [])
        user_query = messages[-1].content if messages else ""
        additional_material = state.get("additional_material", [])
        
        effective_memory = state.get("effective_memory")
        recent_msgs_for_qu = state.get("recent_msgs_for_qu", [])

        # Emit a tool call event to show the planning step in the UI
        await adispatch_custom_event(
            "manually_emit_tool_call",  # An AG-UI event to trigger tool call visualization
            data={
                "id": str(uuid.uuid4()),
                "name": "understand_user_message",
                "args": {}
            },
            config=config
        )
        await asyncio.sleep(0.1)  

        # Always run QueryUnderstanding — handles jargon, abbreviations, pronouns
        qu_output = await self.query_understanding.run(
            user_query=user_query,
            session_memory=effective_memory,
            recent_messages=recent_msgs_for_qu,
        )
        logger.info(f"[QU] clarified='{qu_output.clarified_query}' is_ambiguous={qu_output.is_ambiguous}")

        # Short-circuit: if query is ambiguous, ask for clarification immediately
        # Skip if user attached images/videos — visual context resolves the ambiguity
        if qu_output.is_ambiguous and qu_output.clarifying_questions and not additional_material:
            questions_text = "\n".join(f"- {q}" for q in qu_output.clarifying_questions)
            clarification_response = AIMessage(content=(
                f"Câu hỏi của bạn chưa đủ rõ ràng để tôi trả lời chính xác. "
                f"Bạn có thể làm rõ thêm không?\n{questions_text}"
            ))
            logger.info("[QU] is_ambiguous=True — returning clarification request, skipping graph.")
            
            return Command(
                goto=END,
                update={
                    "messages": messages + [clarification_response],
                },
            )

        # Update tracing metadata
        langfuse = get_client()
        langfuse.update_current_span(
            metadata={
                "is_ambigous": qu_output.is_ambiguous
            }
        )

        return {
            "messages": state.get("messages", []), # Copilotkit will append new user messages to "messages"
            "claried_query": qu_output.clarified_query,
        }
