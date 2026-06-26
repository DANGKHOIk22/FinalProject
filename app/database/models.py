from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database.db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    phone_number = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)

    # Relationship: one user can have many threads (flexible design)
    threads = relationship("UserThread", back_populates="user", lazy="dynamic")


class UserThread(Base):
    """Maps user_id to thread_id for conversation history persistence.
    
    Flexible design: user_id is NOT unique, so one user can own multiple
    threads in the future. For now, the API returns the most recently
    updated active thread (or creates one).
    """
    __tablename__ = "user_threads"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    thread_id = Column(String, unique=True, nullable=False, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationship back to User
    user = relationship("User", back_populates="threads")
