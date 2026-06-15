"""
Test case để kiểm tra kết nối memory, thêm và lấy dữ liệu.

Yêu cầu một PostgreSQL instance đang chạy và cấu hình qua POSTGRES_DATABASE_URL.
Test sẽ tự động bị skip nếu DB không sẵn sàng (ví dụ trong CI mặc định) — local
dev có DB thì chạy bình thường.
"""
import logging
import os
import uuid
import asyncio

import psycopg
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory

from app.soccer_agent.memory.chat_history import get_connection_pool, ConversationHistoryManager

logger = logging.getLogger(__name__)


def _postgres_available() -> bool:
    """Quick probe: open a real connection with a short timeout. Returns False
    when the DB is unreachable so the test can be skipped instead of failing."""
    dsn = os.getenv("POSTGRES_DATABASE_URL")
    if not dsn:
        return False
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn:  # type: ignore[arg-type]
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.info(f"Postgres probe failed → skipping memory tests: {e}")
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="POSTGRES_DATABASE_URL not set or PostgreSQL not reachable",
)


def ensure_table_exists(connection, table_name: str = "messages_agents"):
    """
    Đảm bảo bảng messages_agents tồn tại trong database.
    Sử dụng PostgresChatMessageHistory.create_tables() để tạo bảng.
    """
    try:
        # Sử dụng method create_tables của PostgresChatMessageHistory
        # Tạo một instance tạm để gọi create_tables
        temp_history = PostgresChatMessageHistory(
            table_name=table_name,
            session_id=str(uuid.uuid4()),  # Phải là UUID hợp lệ
            sync_connection=connection
        )
        # Gọi create_tables để tạo bảng nếu chưa có
        temp_history.create_tables()
        connection.commit()
        logger.info(f"✅ Đã đảm bảo bảng {table_name} tồn tại!")
    except Exception as e:
        logger.warning(f"Không thể tạo bảng tự động, thử tạo thủ công: {e}")
        # Fallback: tạo bảng thủ công
        try:
            with connection.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR NOT NULL,
                        message JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_session_id ON {table_name}(session_id);
                """)
                connection.commit()
                logger.info(f"✅ Đã tạo bảng {table_name} thủ công thành công!")
        except Exception as e2:
            logger.error(f"Lỗi khi tạo bảng: {e2}")
            connection.rollback()
            raise


@pytest.mark.asyncio
async def test_memory_connection_add_and_get():
    """
    Test case kiểm tra:
    1. Kết nối đến PostgreSQL memory
    2. Thêm dữ liệu vào memory bằng ConversationHistoryManager
    3. Lấy dữ liệu từ memory
    4. Xác minh dữ liệu đã được lưu đúng
    5. Xóa dữ liệu và kiểm tra
    """
    # Tạo session_id ngẫu nhiên (phải là UUID)
    session_id = str(uuid.uuid4())
    logger.info(f"Testing với session_id: {session_id}")
    
    # Dữ liệu test
    test_user_query = "Ai là cầu thủ ghi bàn nhiều nhất trong lịch sử World Cup?"
    test_agent_response = "Cầu thủ ghi bàn nhiều nhất trong lịch sử World Cup là Miroslav Klose với 16 bàn thắng."
    
    try:
        # 0. Đảm bảo bảng tồn tại trước
        logger.info("Bước 0: Đảm bảo bảng messages_agents tồn tại...")
        pool = get_connection_pool()
        temp_conn = pool.getconn()
        try:
            ensure_table_exists(temp_conn)
        finally:
            pool.putconn(temp_conn)
        
        # 1. Test kết nối và tạo ConversationHistoryManager
        logger.info("Bước 1: Khởi tạo ConversationHistoryManager...")
        manager = ConversationHistoryManager(session_id=session_id)
        
        # Kiểm tra khởi tạo thành công
        assert manager is not None
        assert manager.session_id == session_id
        logger.info("✅ Khởi tạo thành công!")
        
        # 2. Test thêm dữ liệu vào memory
        logger.info("Bước 2: Thêm dữ liệu vào memory...")
        user_msg = HumanMessage(content=test_user_query)
        ai_msg = AIMessage(content=test_agent_response)
        
        await asyncio.to_thread(manager.save_messages, user_msg, ai_msg)
        logger.info("✅ Đã thêm dữ liệu vào memory!")
        
        # 3. Test lấy dữ liệu từ memory
        logger.info("Bước 3: Lấy dữ liệu từ memory...")
        retrieved_history = await asyncio.to_thread(manager.load_messages)
        
        # Kiểm tra dữ liệu đã được lưu
        assert retrieved_history is not None, "Không thể lấy history từ memory"
        assert len(retrieved_history) == 2, f"History phải có 2 messages, nhưng có {len(retrieved_history)}"
        logger.info(f"✅ Đã lấy được {len(retrieved_history)} messages từ memory")
        
        # 4. Xác minh nội dung dữ liệu
        logger.info("Bước 4: Xác minh nội dung dữ liệu...")
        
        user_messages = [msg for msg in retrieved_history if isinstance(msg, HumanMessage)]
        ai_messages = [msg for msg in retrieved_history if isinstance(msg, AIMessage)]
        
        assert len(user_messages) == 1, "Không tìm thấy user message"
        assert len(ai_messages) == 1, "Không tìm thấy assistant message"
        
        assert user_messages[0].content == test_user_query
        assert ai_messages[0].content == test_agent_response
        logger.info("✅ Cả 2 messages đã được lưu đúng nội dung!")
        
        # 5. Test thêm nhiều messages
        logger.info("Bước 5: Test thêm nhiều messages...")
        additional_queries = [
            "Trận đấu nào có nhiều bàn thắng nhất trong lịch sử World Cup?",
            "Ai là đội vô địch World Cup 2022?"
        ]
        additional_responses = [
            "Trận đấu có nhiều bàn thắng nhất là Áo vs Thụy Sĩ (7-5) năm 1954.",
            "Đội vô địch World Cup 2022 là Argentina."
        ]
        
        for query, response in zip(additional_queries, additional_responses):
            await asyncio.to_thread(
                manager.save_messages,
                HumanMessage(content=query),
                AIMessage(content=response)
            )
        
        # Lấy lại history sau khi thêm nhiều messages
        updated_history = await asyncio.to_thread(manager.load_messages)
        assert len(updated_history) == 6, f"Expected 6 messages, got {len(updated_history)}"
        logger.info(f"✅ Đã thêm thành công {len(additional_queries)} cặp messages mới!")
        
        # 6. Test clear history
        logger.info("Bước 6: Test clear history...")
        await asyncio.to_thread(manager.clear)
        cleared_history = await asyncio.to_thread(manager.load_messages)
        assert len(cleared_history) == 0, f"History phải trống sau khi clear, nhưng có {len(cleared_history)}"
        logger.info("✅ Đã clear history thành công!")
        
        logger.info("=" * 70)
        logger.info("✅ TẤT CẢ CÁC TEST ĐÃ PASS!")
        logger.info(f"Session ID: {session_id}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình test: {e}", exc_info=True)
        raise
