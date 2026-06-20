from typing import Annotated
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.database.db import get_db
from app.database import models
from app.schema.user_schema import UserCreate, UserLogin, AuthResponse, ThreadResponse
from app.api.deps import create_access_token, get_password_hash, verify_password, get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/register", response_model=AuthResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.phone_number == user.phone_number).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Phone number already registered")
    hashed_password = get_password_hash(user.password)
    new_user = models.User(full_name=user.full_name, phone_number=user.phone_number, password_hash=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = create_access_token(data={"sub": new_user.phone_number, "id": new_user.id})
    return {"access_token": access_token, "token_type": "bearer", "user": new_user}

@router.post("/login", response_model=AuthResponse)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.phone_number == user.phone_number).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid phone number or password")
    
    access_token = create_access_token(data={"sub": db_user.phone_number, "id": db_user.id})
    return {"access_token": access_token, "token_type": "bearer", "user": db_user}


@router.get("/thread", response_model=ThreadResponse)
def get_or_create_thread(
    current_user: Annotated[models.User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Lấy hoặc tạo thread_id cho user hiện tại.

    Trả về thread đang active gần nhất. Nếu chưa có thread nào,
    tự động tạo mới một thread_id và lưu vào bảng `user_threads`.

    Thiết kế linh hoạt: bảng `user_threads` hỗ trợ nhiều thread
    trên một user (user_id không unique). Hiện tại API luôn trả về
    thread active mới nhất.
    """
    # Tìm thread active gần nhất của user
    user_thread = (
        db.query(models.UserThread)
        .filter(
            models.UserThread.user_id == current_user.id,
            models.UserThread.is_active is True,
        )
        .order_by(models.UserThread.updated_at.desc())
        .first()
    )

    if user_thread:
        return ThreadResponse(thread_id=user_thread.thread_id)

    # Chưa có thread nào — tạo mới với retry chống race condition
    new_thread_id = str(uuid.uuid4())
    new_thread = models.UserThread(
        user_id=current_user.id,
        thread_id=new_thread_id,
    )
    db.add(new_thread)
    try:
        db.commit()
        db.refresh(new_thread)
        logger.info(
            f"Created new thread {new_thread_id} for user {current_user.id}"
        )
        return ThreadResponse(thread_id=new_thread.thread_id)
    except IntegrityError:
        db.rollback()
        # Một request song song đã tạo thread trước — lấy lại thread đó
        user_thread = (
            db.query(models.UserThread)
            .filter(
                models.UserThread.user_id == current_user.id,
            )
            .order_by(models.UserThread.updated_at.desc())
            .first()
        )
        if user_thread:
            return ThreadResponse(thread_id=user_thread.thread_id)
        # Fallback cực kỳ hiếm: retry tạo mới một lần nữa
        new_thread_id = str(uuid.uuid4())
        new_thread = models.UserThread(
            user_id=current_user.id,
            thread_id=new_thread_id,
        )
        db.add(new_thread)
        db.commit()
        db.refresh(new_thread)
        return ThreadResponse(thread_id=new_thread.thread_id)
