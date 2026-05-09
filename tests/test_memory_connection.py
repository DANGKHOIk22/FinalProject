"""
Test case để kiểm tra kết nối memory, thêm và lấy dữ liệu.

Yêu cầu một PostgreSQL instance đang chạy và cấu hình qua POSTGRES_DATABASE_URL.
Test sẽ tự động bị skip nếu DB không sẵn sàng (ví dụ trong CI mặc định) — local
dev có DB thì chạy bình thường.
"""
import logging
import os
import uuid

import psycopg
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory

from app.soccer_agent.memory.chat_history import get_connection_pool, get_postgres_memory
from app.soccer_agent.memory.conversation_memory import CustomSystemPromptMemory

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
            session_id="temp",
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
    2. Thêm dữ liệu vào memory
    3. Lấy dữ liệu từ memory
    4. Xác minh dữ liệu đã được lưu đúng
    """
    # Tạo user_id ngẫu nhiên
    user_id = str(uuid.uuid4())
    logger.info(f"Testing với user_id: {user_id}")
    
    # Dữ liệu test
    test_user_query = "Ai là cầu thủ ghi bàn nhiều nhất trong lịch sử World Cup?"
    test_agent_response = "Cầu thủ ghi bàn nhiều nhất trong lịch sử World Cup là Miroslav Klose với 16 bàn thắng."
    
    connection = None
    pool = None
    memory_object = None
    
    try:
        # 0. Đảm bảo bảng tồn tại trước
        logger.info("Bước 0: Đảm bảo bảng messages_agents tồn tại...")
        pool = get_connection_pool()
        temp_conn = pool.getconn()
        try:
            ensure_table_exists(temp_conn)
        finally:
            pool.putconn(temp_conn)
        
        # 1. Test kết nối và tạo memory object
        logger.info("Bước 1: Kiểm tra kết nối đến PostgreSQL memory...")
        memory, connection, pool = get_postgres_memory(user_id)
        
        # Tạo CustomSystemPromptMemory wrapper
        memory_object = CustomSystemPromptMemory(
            memory_key="history",
            chat_memory=memory.chat_memory,
            return_messages=True,
            max_history=15,
        )
        
        # Kiểm tra kết nối thành công
        assert memory is not None, "Memory object không được tạo"
        assert connection is not None, "Connection không được tạo"
        assert pool is not None, "Pool không được tạo"
        assert memory_object is not None, "Memory object wrapper không được tạo"
        logger.info("✅ Kết nối thành công!")
        
        # 2. Test thêm dữ liệu vào memory
        logger.info("Bước 2: Thêm dữ liệu vào memory...")
        memory_object.save_context(
            inputs={"input": test_user_query},
            outputs={"output": test_agent_response}
        )
        logger.info("✅ Đã thêm dữ liệu vào memory!")
        
        # 3. Test lấy dữ liệu từ memory
        logger.info("Bước 3: Lấy dữ liệu từ memory...")
        history = memory_object.load_memory_variables({})
        retrieved_history = history.get("history", [])
        
        # Kiểm tra dữ liệu đã được lưu
        assert retrieved_history is not None, "Không thể lấy history từ memory"
        assert len(retrieved_history) > 0, "History rỗng sau khi thêm dữ liệu"
        logger.info(f"✅ Đã lấy được {len(retrieved_history)} messages từ memory")
        
        # 4. Xác minh nội dung dữ liệu
        logger.info("Bước 4: Xác minh nội dung dữ liệu...")
        
        # Tìm user message và assistant message
        user_messages = [msg for msg in retrieved_history if isinstance(msg, HumanMessage)]
        ai_messages = [msg for msg in retrieved_history if isinstance(msg, AIMessage)]
        
        assert len(user_messages) > 0, "Không tìm thấy user message"
        assert len(ai_messages) > 0, "Không tìm thấy assistant message"
        
        # Kiểm tra nội dung user query
        found_user_query = False
        for msg in user_messages:
            if test_user_query in msg.content:
                found_user_query = True
                break
        assert found_user_query, f"Không tìm thấy user query: {test_user_query}"
        logger.info("✅ User query đã được lưu đúng!")
        
        # Kiểm tra nội dung agent response
        found_agent_response = False
        for msg in ai_messages:
            if isinstance(msg.content, str) and test_agent_response in msg.content:
                found_agent_response = True
                break
            elif isinstance(msg.content, list):
                # Nếu content là list (có thể có thought signatures)
                for content_item in msg.content:
                    if isinstance(content_item, str) and test_agent_response in content_item:
                        found_agent_response = True
                        break
        assert found_agent_response, f"Không tìm thấy agent response: {test_agent_response}"
        logger.info("✅ Agent response đã được lưu đúng!")
        
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
            memory_object.save_context(
                inputs={"input": query},
                outputs={"output": response}
            )
        
        # Lấy lại history sau khi thêm nhiều messages
        updated_history = memory_object.load_memory_variables({})
        updated_messages = updated_history.get("history", [])
        
        assert len(updated_messages) >= len(retrieved_history) + len(additional_queries) * 2, \
            f"Không đủ messages sau khi thêm. Expected >= {len(retrieved_history) + len(additional_queries) * 2}, got {len(updated_messages)}"
        logger.info(f"✅ Đã thêm thành công {len(additional_queries)} cặp messages mới!")
        
        logger.info("=" * 70)
        logger.info("✅ TẤT CẢ CÁC TEST ĐÃ PASS!")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Tổng số messages trong memory: {len(updated_messages)}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình test: {e}", exc_info=True)
        raise
    finally:
        # Cleanup: Trả connection về pool
        if connection and pool:
            try:
                # Commit any pending transactions
                if not getattr(connection, "closed", False):
                    connection.commit()
                
                # Return connection to pool
                pool.putconn(connection)
                logger.info("✅ Đã cleanup connection thành công!")
            except Exception as cleanup_error:
                logger.warning(f"Lỗi trong quá trình cleanup: {cleanup_error}")
                # Fallback: close connection if pool return fails
                try:
                    if connection and not getattr(connection, "closed", False):
                        connection.close()
                except Exception:
                    pass


if __name__ == "__main__":
    import asyncio
    import sys
    
    # Cấu hình logging để hiển thị output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Chạy test trực tiếp
    try:
        asyncio.run(test_memory_connection_add_and_get())
        print("\n✅ Test completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

