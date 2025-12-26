"""
Custom Memory Wrapper for Conversation History

Extends ConversationBufferMemory with:
- History trimming (limit to max_history messages)
- Safe rollback on errors
- Retry mechanisms for save/load operations
"""
import logging
from typing import List, Dict, Any
from pydantic import Field
from langchain_classic.memory.buffer import ConversationBufferMemory
from langchain_core.messages import BaseMessage, SystemMessage

logger = logging.getLogger(__name__)


class CustomSystemPromptMemory(ConversationBufferMemory):
    """
    Custom memory wrapper with enhanced error handling and history management.
    
    Features:
    - Trims history to max_history messages (prevents token overflow)
    - Safe rollback on database errors
    - Retry mechanisms for save/load operations
    - Graceful degradation (returns system prompt if all else fails)
    """

    system_prompt: str = Field(default="You are a helpful assistant.")
    max_history: int = Field(default=15)

    def __init__(self, max_history: int = 10, **kwargs):
        """
        Initialize CustomSystemPromptMemory.
        
        Args:
            max_history: Maximum number of messages to keep in history
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            max_history=max_history,
            **kwargs
        )

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables with trimmed history.
        
        Trims history to max_history messages to prevent token overflow
        and improve performance. Includes retry mechanism on errors.
        
        Args:
            inputs: Input dictionary (not used, but required by interface)
            
        Returns:
            Dict with memory_key containing list of trimmed messages
            Falls back to system prompt if all else fails
        """
        try:
            history: List[BaseMessage] = self.chat_memory.messages
            # Trim to last max_history messages
            trimmed_history = (
                history[-self.max_history:] 
                if len(history) > self.max_history 
                else history
            )
            return {self.memory_key: trimmed_history}

        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            # Try rollback and retry
            self._safe_rollback()
            
            try:
                history: List[BaseMessage] = self.chat_memory.messages
                trimmed_history = (
                    history[-self.max_history:] 
                    if len(history) > self.max_history 
                    else history
                )
                return {self.memory_key: trimmed_history}
            except Exception:
                # Final fallback: return system prompt only
                logger.warning("Returning system prompt only")
                return {self.memory_key: [SystemMessage(content=self.system_prompt)]}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save conversation context with error handling and retry.
        
        Args:
            inputs: Input messages (typically {"input": user_query})
            outputs: Output messages (typically {"output": assistant_response})
            
        Note:
            Includes retry mechanism - attempts save twice before giving up.
        """
        try:
            super().save_context(inputs, outputs)
            logger.debug("Context saved successfully")
        except Exception as e:
            logger.error(f"Error saving context: {e}")
            self._safe_rollback()
            
            # Retry once after rollback
            try:
                super().save_context(inputs, outputs)
                logger.debug("Context saved on retry")
            except Exception as e2:
                logger.error(f"Failed to save context after retry: {e2}")

    def _safe_rollback(self):
        """
        Safely rollback database connection.
        
        This method attempts to rollback any pending transactions and
        clear prepared statements. Errors are logged but not raised.
        """
        try:
            # Access connection from chat_memory (PostgresChatMessageHistory)
            if hasattr(self.chat_memory, '_connection'):
                conn = self.chat_memory._connection
                if conn and not conn.closed:
                    conn.rollback()
                    # Clear prepared statements to avoid conflicts
                    if hasattr(conn, 'prepare_threshold'):
                        conn.prepare_threshold = None
                    logging.debug("Connection rolled back")
        except Exception as e:
            logging.debug(f"Rollback error (ignored): {e}")

    def clear(self) -> None:
        """
        Clear all conversation history.
        
        Safely clears memory after rolling back any pending transactions.
        """
        try:
            self._safe_rollback()
            self.chat_memory.clear()
            logger.info("Memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")