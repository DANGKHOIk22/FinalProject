from app.database.db import engine
from app.database.models import Base

# Tạo bảng trong database nếu chưa có
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")
