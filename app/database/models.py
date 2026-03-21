from sqlalchemy import Column, Integer, String
from app.database.db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    phone_number = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
