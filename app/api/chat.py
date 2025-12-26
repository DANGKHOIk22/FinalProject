import os
import logging
import uvicorn
from fastapi import APIRouter, HTTPException,Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from app.soccer_agent.agent import get_agent_service
from app.schema.chat import ChatRequest
logger = logging.getLogger(__name__)
# --- FastAPI App ---
router = APIRouter()

# --- Endpoint /chat ---
@router.post("/chat")
async def chat_endpoint(request: ChatRequest, soccer_agent=Depends(get_agent_service)):
    """
    Xử lý câu hỏi của người dùng bằng SoccerAgent (Planning -> Execution).
    """
    if soccer_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Soccer Agent is not running. Check server logs for initialization errors."
        )

    try:
        # Gọi phương thức run của Agent
        final_answer_content = soccer_agent.run(
            request=request
        )
        
        # Hàm run() của bạn hiện tại trả về content của ToolMessage cuối cùng.
        return {
            "status": "success",
            "user_query": request.user_query,
            "agent_response": final_answer_content
        }

    except Exception as e:
        # Xử lý các lỗi xảy ra trong quá trình Agent thực thi
        logger.error(f"Critical error during agent execution: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during agent execution: {str(e)}"
        )