import os
import psycopg
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InitDB")

def init_db():
    # Lấy Database URL từ environment
    db_url = os.getenv("POSTGRES_DATABASE_URL")
    
    if not db_url:
        logger.error("No database URL found in environment variables (POSTGRES_URL or DATABASE_URL).")
        return

    sql_commands = [
        # 1. Kích hoạt extension vector
        "CREATE EXTENSION IF NOT EXISTS vector;",
        
        # 2. Tạo bảng lưu trữ Memory
        """
        CREATE TABLE IF NOT EXISTS user_long_term_memory (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id TEXT NOT NULL,
            entity_name TEXT,
            content TEXT NOT NULL,
            metadata JSONB,
            embedding vector(768),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        
        # 3. Tạo index HNSW cho vector search
        "CREATE INDEX IF NOT EXISTS idx_memory_embedding_hnsw ON user_long_term_memory USING hnsw (embedding vector_cosine_ops);",
        
        # 4. Tạo index cho filter
        "CREATE INDEX IF NOT EXISTS idx_memory_user_entity ON user_long_term_memory (user_id, entity_name);",
        "CREATE INDEX IF NOT EXISTS idx_memory_created_at ON user_long_term_memory (created_at);"
    ]

    try:
        logger.info("Connecting to Supabase Postgres...")
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for cmd in sql_commands:
                    cmd_summary = cmd.strip().split('\n')[0][:50] + "..."
                    logger.info(f"Executing: {cmd_summary}")
                    cur.execute(cmd)
                conn.commit()
        logger.info("✅ Database initialisation completed successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialise database: {e}")

if __name__ == "__main__":
    init_db()
