import logging
import uuid
import asyncio
from typing import Any
from langfuse import get_client

from langgraph.graph.state import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks.manager import adispatch_custom_event
from app.schema.soccer_agent.state import AgentState
from app.soccer_agent.memory.chat_history import ConversationHistoryManager
from app.services.media_registry import MediaRegistryService
logger = logging.getLogger(__name__)

class ConversationHistoryNode:
    def __init__(self):
        pass

    async def get_conversational_history(self, state: AgentState, config: RunnableConfig):
        """
        This node retrieves the conversation history from the storage and updates the state.
        """
        thread_id = config.get("configurable", {}).get("thread_id") or str(uuid.uuid4())
        user_id = str(config.get("configurable", {}).get("user_id"))
        
        # --- Emit tool call for rendering this step in UI ---
        try:
            await adispatch_custom_event(
                "manually_emit_tool_call",
                data={
                    "id": str(uuid.uuid4()),
                    "name": "understand_user_message",
                    "args": {}
                },
                config=config
            )
        except RuntimeError as e:
            logger.warning(
                f"Failed to dispatch custom event: {e}. "
                "This is expected if running outside a LangChain/LangGraph run context (e.g., in unit tests)."
            )
        # --- Flags for tracing metadata ---
        loading_history_status = "failed" # "success"/"failed"
        load_from_cache = False # True: This question has already stored in cache, just load from cache database
        pass_full_history = False # True: The history length is not over threshold limit, give the agent full history without summary

        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="chain", 
            name="create_history_string") as observation:

            # Load conversation history
            chat_history = None
            if thread_id:
                try:
                    manager = ConversationHistoryManager(session_id=thread_id)
                    chat_history = await asyncio.to_thread(manager.load_messages)
                    loading_history_status = "success"
                except Exception as e:
                    logging.warning(f"Memory load failed for session {thread_id}: {e}")

            # Build history
            history_text = ""
            recent_msgs = []
            if chat_history:
                # Filter to only keep real chat turns (Human and AI)
                # EXCLUDE: ToolMessage, and AIMessage that are just tool calls (no content or has tool_calls)
                filtered_history = []
                for msg in chat_history:
                    msg_text = msg.text
                    if isinstance(msg, HumanMessage) and msg_text.strip():
                        filtered_history.append(msg)
                    elif isinstance(msg, AIMessage):
                        # Only keep AI messages that have actual text and are NOT tool calls
                        has_tool_calls = hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0
                        if msg_text.strip() and not has_tool_calls:
                            filtered_history.append(msg)
            
                # Trim messages 
                filtered_history = filtered_history[-10:]
                
                history_text = "### LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n"
                # Build history text
                for msg in filtered_history:
                    if isinstance(msg, HumanMessage):
                        history_text += f"{ConversationHistoryNode.create_user_message_string(msg)}\n"
                    elif isinstance(msg, AIMessage):
                        history_text += f"Assistant: {msg.text}\n"
                pass_full_history = True
                
                logger.info(f"[History] Retrieved {len(recent_msgs)} clean chat turns.")

            observation.update(
                output={
                    "chat_history": chat_history or [],
                    "history_text": history_text or "No conversation history."
                }
            )

        # --- Update tracing span metadata ---
        langfuse.update_current_span(
            metadata={
                "loading_history_status": loading_history_status,
                "load_from_cache": load_from_cache,
                "pass_full_history": pass_full_history,
            }
        )
        
        return {
            "conversation_history": history_text if history_text else "No conversation history.",
            "parallel_results": [],
            "tool_calls_history": [],
            "tool_results_history": []
        }

    @staticmethod
    def create_user_message_string(message: HumanMessage) -> str:
        """
        Helper function to safely create a string representation of a HumanMessage for history.
        It will add media id to the messages str
        """
        msg_str = f"User: {message.text}"
        media = MediaRegistryService.extract_uuids_from_message(message)
        image_ids = media.get("image_ids", [])
        video_id = media.get("video_id")
        
        if image_ids:
            msg_str += f"\n\tAttached image: {', '.join(image_ids)}"
        if video_id:
            msg_str += f"\n\tAttached video: {video_id}"
            
        return msg_str
