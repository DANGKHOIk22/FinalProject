"""
PostgreSQL Memory Management for Chat History

This module provides connection pooling and memory management for storing
conversation history in PostgreSQL. Uses psycopg connection pool for
efficient connection reuse.
"""
import logging
from psycopg_pool import ConnectionPool
from psycopg import Connection
from langchain_classic.memory.buffer import ConversationBufferMemory
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory

from app.config import settings

# Global connection pool (singleton pattern)
_connection_pool = None


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


def get_postgres_memory(session_id: str) -> tuple:
    """
    Create memory instance with connection from pool.
    
    This function:
    1. Gets connection from pool
    2. Ensures clean connection state (rollback, disable prepared statements)
    3. Creates PostgresChatMessageHistory with the connection
    4. Wraps it in ConversationBufferMemory
    
    Args:
        session_id: Session ID for conversation history
        
    Returns:
        Tuple of (memory, connection, pool):
            - memory: ConversationBufferMemory instance
            - connection: Database connection (for cleanup)
            - pool: Connection pool (for cleanup)
            
    Note:
        Caller is responsible for cleanup (returning connection to pool).
        See agent cleanup() methods.
    """
    pool = get_connection_pool()
    
    # Get connection from pool
    sync_conn = pool.getconn()
    
    try:
        # Ensure clean state: rollback any pending transactions
        sync_conn.rollback()
        
        # Disable prepared statements to avoid conflicts
        sync_conn.prepare_threshold = None
        
    except Exception as e:
        logging.warning(f"Connection cleanup error: {e}")
        # If connection is bad, close it and get a new one
        try:
            sync_conn.close()
        except Exception:
            pass
        sync_conn = pool.getconn()
        sync_conn.prepare_threshold = None
    
    # Create message history with the connection
    message_history = PostgresChatMessageHistory(
        "messages_agents",  # Table name for storing messages
        session_id,          # Session ID for conversation isolation
        sync_connection=sync_conn
    )
    
    # Wrap in ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True  # Return as message objects, not strings
    )
    
    return memory, sync_conn, pool