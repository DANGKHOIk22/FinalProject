from contextlib import asynccontextmanager
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import cấu hình và database
from app.config.settings import settings

# Chỉ import router chat
from app.api.chat import router as chat_router

# Cấu hình logging đơn giản thay vì structlog
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 1. Lifespan: Quản lý khởi động/tắt app ---
# --- 2. Khởi tạo FastAPI App ---
app = FastAPI(
    title="Soccer Knowledge Agent API",
    description="API for the LangGraph-based Soccer Agent for tool chain execution.",
    version="1.0.0",
)

# --- 3. Cấu hình CORS ---
# Cho phép Frontend gọi vào API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --- 4. Đăng ký Router ---
# Chỉ đăng ký duy nhất chat router
app.include_router(chat_router, tags=["Soccer Chat Agent"])

# --- 5. Root Endpoint (Health check) ---
@app.get("/")
def root():
    return {
        "message": "Soccer Agent API is running!", 
        "docs": "/docs",
    }

# --- 6. Chạy server ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )