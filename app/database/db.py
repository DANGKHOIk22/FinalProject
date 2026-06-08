from sqlalchemy import NullPool, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config.settings import settings

# Kết nối tới Postgres
DATABASE_URL = settings.POSTGRES_DATABASE_URL

# Because SQLAlchemy 2.0 requires the use of a specific driver for PostgreSQL, we need to ensure that the connection string is in the correct format. If the connection string starts with "postgresql://", we replace it with "postgresql+psycopg://" to specify that we want to use the psycopg driver.
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(DATABASE_URL, 
                       poolclass=NullPool, 
                       pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
db = get_db()