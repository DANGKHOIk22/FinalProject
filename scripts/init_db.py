from app.database.db import engine
from app.database.models import Base
from app.config.settings import settings
import psycopg

# Tạo bảng trong database nếu chưa có
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("SQLAlchemy tables created.")

    # Tạo bảng session_memories cho Session Memory Manager
    with psycopg.connect(settings.POSTGRES_DATABASE_URL) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_memories (
                session_id  VARCHAR PRIMARY KEY,
                memory_json TEXT NOT NULL,
                updated_at  TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
    print("session_memories table created.")
