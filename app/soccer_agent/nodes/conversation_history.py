import logging
import uuid
import asyncio
from langfuse import get_client

from langgraph.graph.state import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_core.callbacks.manager import adispatch_custom_event
from app.schema.soccer_agent.state import AgentState
from app.soccer_agent.memory.chat_history import get_postgres_memory
from app.soccer_agent.memory.conversation_memory import CustomSystemPromptMemory

logger = logging.getLogger(__name__)

class ConversationHistoryNode:
    def __init__(self):
        pass

    async def get_memory(self, session_id: str):
        """Get conversation memory for the given session."""
        memory, connection, pool = get_postgres_memory(session_id)
        memory_object = CustomSystemPromptMemory(
            memory_key="history",
            chat_memory=memory.chat_memory,
            return_messages=True,
            max_history=15,  # Limit to last 15 messages
        )
        return memory_object, connection, pool
    
    async def cleanup(self, connection, pool, session_id: str):
        """Cleanup database connection and return to pool."""
        try:
            if not connection:
                return

            # Try to commit any pending transactions
            try:
                closed = getattr(connection, "closed", False)
                if not closed:
                    connection.commit()
            except Exception:
                # Best-effort commit; ignore errors here
                pass

            # Return connection to pool if available, otherwise close it
            if pool:
                try:
                    pool.putconn(connection)
                    logging.debug(f"Connection returned to pool for session {session_id}")
                except Exception:
                    try:
                        connection.close()
                    except Exception:
                        pass
            else:
                try:
                    connection.close()
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
            try:
                if connection:
                    connection.close()
            except Exception:
                pass

    async def get_conversational_history(self, state: AgentState, config: RunnableConfig):
        """
        This node retrieves the conversation history from the storage and updates the state.
        """
        metadata = config.get("metadata", {})
        thread_id = metadata.get("thread_id", str(uuid.uuid4()))
        user_id = config.get("configurable", {}).get("user_id")
        
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
                    memory_object, connection, pool = await self.get_memory(session_id=thread_id)
                    try:
                        history = memory_object.load_memory_variables({})
                        chat_history = history.get("history")
                        loading_history_status = "success"
                    except Exception as e:
                        logging.warning(f"Memory load failed for session {thread_id}: {e}")
                    finally:
                        await self.cleanup(connection=connection, pool=pool, session_id=thread_id)
                except Exception as e:
                    logging.warning(f"Failed to obtain memory for session {thread_id}: {e}")

            # Build history
            history_text = ""
            recent_msgs_for_qu = []

            if chat_history:
                # Filter to only keep real chat turns (Human and AI)
                # EXCLUDE: ToolMessage, and AIMessage that are just tool calls (no content or has tool_calls)
                filtered_history = []
                for msg in chat_history:
                    msg_type = msg.__class__.__name__
                    content = msg.content if isinstance(msg.content, str) else ""
                    
                    if msg_type == "HumanMessage" and content.strip():
                        filtered_history.append(msg)
                    elif msg_type == "AIMessage":
                        # Only keep AI messages that have actual text and are NOT tool calls
                        has_tool_calls = hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0
                        if content.strip() and not has_tool_calls:
                            filtered_history.append(msg)
                
                # Take last 10-15 messages for short-term context
                recent_msgs_for_qu = filtered_history[-10:]
                
                if recent_msgs_for_qu:
                    raw_history_text = "### LỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n"
                    for msg in recent_msgs_for_qu:
                        role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                        raw_history_text += f"{role}: {msg.content}\n"
                    history_text = raw_history_text
                    pass_full_history = True
                
                logger.info(f"[History] Retrieved {len(recent_msgs_for_qu)} clean chat turns.")

            observation.update(
                output={
                    "chat_history": chat_history or [],
                    "history_text": history_text or "No conversation history.",
                    "recent_msgs_for_qu": recent_msgs_for_qu
                }
            )

            observation.update(
                output={
                    "chat_history": chat_history or "",
                    "history_text": history_text or "No conversation history.",
                    "recent_msgs_for_qu": recent_msgs_for_qu
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
            "conversation_history": history_text,
            "recent_msgs_for_qu": recent_msgs_for_qu,
            "effective_memory": None,
            # Reset history fields to prevent accumulation across turns in the same thread_id
            "parallel_results": [],
            "tool_calls_history": [],
            "tool_results_history": []
        }
