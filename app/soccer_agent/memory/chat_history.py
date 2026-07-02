"""
PostgreSQL Memory Management for Chat History

This module provides connection pooling and memory management for storing
conversation history in PostgreSQL. Uses psycopg connection pool for
efficient connection reuse.
"""
import logging
from typing import List
from psycopg_pool import ConnectionPool
from psycopg import Connection
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory

from app.config import settings

# Global connection pool (singleton pattern)
_connection_pool = None
logger = logging.getLogger(__name__)


def get_connection_pool() -> ConnectionPool:
    """
    Get or create singleton connection pool.
    
    The pool is created once and reused across all requests.
    This significantly improves performance by avoiding connection overhead.
    
    Returns:
        ConnectionPool instance
        
    Configuration:
        - min_size=2: Keep at least 2 connections ready
        - max_size=20: Support up to 20 concurrent connections
        - timeout=30: Wait up to 30s if pool is exhausted
        - prepare_threshold=None: Disable prepared statements (avoids conflicts)
    """
    global _connection_pool
    if _connection_pool is None:
        connection_string = settings.POSTGRES_DATABASE_URL
        
        def configure_connection(conn: Connection):
            """Configure new connections: Disable prepared statements"""
            conn.prepare_threshold = None  # Disable to avoid conflicts
            
        _connection_pool = ConnectionPool(
            connection_string,
            min_size=2,
            max_size=20,  # Support up to 20 concurrent requests
            configure=configure_connection,
            timeout=30  # Timeout if pool is exhausted
        )
    return _connection_pool


class ConversationHistoryManager:
    """
    Interface quản lý tương tác giữa các Node trong Agent và PostgresChatMessageHistory.
    
    Nhiệm vụ:
    - Quản lý kết nối và tự động rollback an toàn khi xảy ra lỗi.
    - Thực hiện ghi/đọc tin nhắn gốc (Content Block) với cơ chế thử lại (retry).
    - Không thực hiện fallback hay trim số lượng tin nhắn (việc trim được xử lý ở các node phía trên).
    """

    def __init__(
        self,
        session_id: str,
        table_name: str = "messages_agents",
        max_history: int = 15,  # Unused, kept for backwards compatibility
        system_prompt: str = "You are a helpful assistant."  # Unused, kept for backwards compatibility
    ):
        self.session_id = session_id
        self.table_name = table_name
        self.pool = get_connection_pool()

    def _safe_rollback(self, conn: Connection):
        """Thực hiện rollback giao dịch an toàn để giải phóng trạng thái lỗi."""
        try:
            if conn and not conn.closed:
                conn.rollback()
                if hasattr(conn, 'prepare_threshold'):
                    conn.prepare_threshold = None
                logger.debug("Database connection rolled back successfully.")
        except Exception as e:
            logger.debug(f"Error during rollback (ignored): {e}")

    def load_messages(self) -> List[BaseMessage]:
        """Tải lịch sử tin nhắn với cơ chế an toàn & thử lại. Lỗi được ném ra thay vì fallback."""
        conn = self.pool.getconn()
        try:
            conn.rollback()
            conn.prepare_threshold = None
            
            history_db = PostgresChatMessageHistory(
                self.table_name,
                self.session_id,
                sync_connection=conn
            )
            messages = history_db.messages
            conn.rollback()  # read-only: close the SELECT's transaction before returning the conn to the pool
            return messages
        except Exception as e:
            logger.warning(f"DB load error: {e}. Attempting rollback and retry...")
            self._safe_rollback(conn)
            try:
                history_db = PostgresChatMessageHistory(
                    self.table_name,
                    self.session_id,
                    sync_connection=conn
                )
                messages = history_db.messages
                conn.rollback()
                return messages
            except Exception as e2:
                logger.error(f"Failed to load memory after retry: {e2}.")
                raise e2
        finally:
            if conn:
                self.pool.putconn(conn)

    def save_messages(self, user_message: BaseMessage, ai_message: BaseMessage) -> None:
        """Ghi nhận câu hỏi và câu trả lời gốc (Content Block) xuống DB một cách an toàn."""
        conn = self.pool.getconn()
        try:
            conn.rollback()
            conn.prepare_threshold = None
            
            history_db = PostgresChatMessageHistory(
                self.table_name,
                self.session_id,
                sync_connection=conn
            )
            # Lưu trực tiếp các đối tượng tin nhắn gốc, bảo toàn toàn bộ Content Block (chứa image_id)
            history_db.add_messages([user_message, ai_message])
            conn.commit()
        except Exception as e:
            logger.warning(f"DB save error: {e}. Attempting rollback and retry...")
            self._safe_rollback(conn)
            try:
                history_db = PostgresChatMessageHistory(
                    self.table_name,
                    self.session_id,
                    sync_connection=conn
                )
                history_db.add_messages([user_message, ai_message])
                conn.commit()
            except Exception as e2:
                logger.error(f"Failed to save context after retry: {e2}")
                raise e2
        finally:
            if conn:
                self.pool.putconn(conn)

    def clear(self) -> None:
        """Xóa sạch lịch sử hội thoại của session."""
        conn = self.pool.getconn()
        try:
            conn.rollback()
            conn.prepare_threshold = None
            history_db = PostgresChatMessageHistory(
                self.table_name,
                self.session_id,
                sync_connection=conn
            )
            history_db.clear()
            conn.commit()
        except Exception as e:
            logger.error(f"Clear history error: {e}")
            self._safe_rollback(conn)
            raise e
        finally:
            if conn:
                self.pool.putconn(conn)