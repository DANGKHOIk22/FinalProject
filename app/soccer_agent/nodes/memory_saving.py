import logging
import uuid
import asyncio
from typing import List

from langchain_core.messages import ToolMessage, ToolCall, HumanMessage
from langgraph.graph.state import RunnableConfig

from app.schema.soccer_agent.state import AgentState
from app.soccer_agent.memory.chat_history import get_postgres_memory
from app.soccer_agent.memory.conversation_memory import CustomSystemPromptMemory
from app.soccer_agent.memory.session_memory import SessionMemoryManager

logger = logging.getLogger(__name__)

class SaveToMemoryNode:
    def __init__(self, session_memory_manager: SessionMemoryManager):
        self.session_memory_manager = session_memory_manager

    def _build_tool_summary_for_memory(
        self,
        tool_calls_history: List[ToolCall],
        tool_results_history: List[ToolMessage],
        final_response: str
    ) -> str:
        """
        Build a structured string combining tool call details and final response
        to be saved into conversation memory for a single chat turn.
        """
        if not tool_calls_history:
            return final_response

        parts = ["[Tool Usage]"]
        for i, tool_call in enumerate(tool_calls_history):
            tool_name = tool_call.get("name", "unknown")
            args = tool_call.get("args", {})
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            parts.append(f"  Step {i + 1}: {tool_name}({args_str})")

            if i < len(tool_results_history):
                result_msg = tool_results_history[i]
                if isinstance(result_msg, dict):
                    content = result_msg.get("content", "")
                    artifact = result_msg.get("artifact")
                else:
                    content = result_msg.content
                    artifact = getattr(result_msg, "artifact", None)
                parts.append(f"    Response: {content}")
                if artifact is not None:
                    parts.append(f"    Artifact: {artifact}")

        parts.append(f"\n[Final Response]\n{final_response}")
        return "\n".join(parts)

    async def _background_save_memory(
        self, session_id: str, user_query: str, output_with_tools: str
    ) -> None:
        """Save conversation memory in background with a dedicated DB connection."""
        try:
            mem, conn, pool = get_postgres_memory(session_id)
            mem_obj = CustomSystemPromptMemory(
                memory_key="history",
                chat_memory=mem.chat_memory,
                return_messages=True,
                max_history=15,
            )
            await asyncio.to_thread(
                mem_obj.save_context,
                {"input": user_query},
                {"output": output_with_tools},
            )
            if conn and pool:
                conn.commit()
                pool.putconn(conn)
        except Exception as e:
            logger.warning(f"[BackgroundSave] Failed for session {session_id}: {e}")

    async def save_to_memory_node(self, state: AgentState, config: RunnableConfig) -> dict:
        """Save the final response along with tool call history into conversation memory."""
        tool_calls_history = state.get("tool_calls_history", [])
        tool_results_history = state.get("tool_results_history", [])
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None 
        metadata = config.get("metadata", {})
        thread_id = metadata.get("thread_id", str(uuid.uuid4()))

        # Find the last user message in the messages history
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break

        try:
            output_with_tools = self._build_tool_summary_for_memory(
                tool_calls_history=tool_calls_history,
                tool_results_history=tool_results_history,
                final_response=last_message.content if last_message else "No response generated",
            )
            asyncio.create_task(
                self._background_save_memory(thread_id, last_user_message.content if last_user_message else "No user message found", output_with_tools)
            )
            # Incremental update of Redis SessionMemory if tool calls were made
            if tool_calls_history:
                asyncio.create_task(
                    self.session_memory_manager.background_update(thread_id, output_with_tools)
                )
        except Exception as e:
            logging.warning(f"Failed to prepare memory save for session {thread_id}: {e}")
        
        return {}
