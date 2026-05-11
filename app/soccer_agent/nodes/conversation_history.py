import logging
import uuid
import asyncio
from langfuse import get_client

from langgraph.graph.state import RunnableConfig
from langchain_core.messages import HumanMessage
from app.schema.soccer_agent.state import AgentState
from app.soccer_agent.memory.chat_history import get_postgres_memory
from app.soccer_agent.memory.conversation_memory import CustomSystemPromptMemory
from app.config.config import SESSION_MEMORY_TOKEN_THRESHOLD, SESSION_MEMORY_RECENT_KEEP
from app.soccer_agent.memory.session_memory import SessionMemoryManager, SessionMemory

logger = logging.getLogger(__name__)

class ConversationHistoryNode:
    def __init__(self, session_memory_manager: SessionMemoryManager):
        self.session_memory_manager = session_memory_manager

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
            effective_memory = SessionMemory()
            recent_msgs_for_qu = []

            if chat_history:
                raw_history_text = "\n\n### LỊCH SỬ HỘI THOẠI:\n"
                for msg in chat_history[-10:]:
                    if hasattr(msg, 'content'):
                        role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                        raw_history_text += f"{role}: {msg.content}\n"

                token_count = self.session_memory_manager.count_tokens(raw_history_text)

                if token_count > SESSION_MEMORY_TOKEN_THRESHOLD:
                    # Path B: history exceeds threshold — use SessionMemory + recent slice
                    old_msgs = chat_history[:-(SESSION_MEMORY_RECENT_KEEP-1)]
                    recent_msgs_for_qu = chat_history[-SESSION_MEMORY_RECENT_KEEP:]

                    cached_memory = await self.session_memory_manager.load_cached_memory(thread_id)
                    if cached_memory is None:
                        if old_msgs:
                            asyncio.create_task(
                                self.session_memory_manager.background_summarise(thread_id, old_msgs)
                            )
                    else:
                        effective_memory = cached_memory
                        load_from_cache = True

                    history_text = self.session_memory_manager.format_compressed_history(
                        effective_memory, recent_msgs_for_qu
                    )
                    pass_full_history = False
                    logger.info(f"[Path B] token_count={token_count}.")
                else:
                    # Path A: full history fits — pass all as recent context
                    recent_msgs_for_qu = chat_history
                    history_text = raw_history_text
                    logger.info(f"[Path A] token_count={token_count}.")
                    pass_full_history = True

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
            "effective_memory": effective_memory
        }
