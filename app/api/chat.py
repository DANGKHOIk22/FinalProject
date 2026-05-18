import logging
from fastapi import APIRouter, HTTPException,Depends
from app.soccer_agent.agent import get_agent_service
from app.schema.chat import ChatRequest
from ag_ui.encoder import EventEncoder
from ag_ui.core.types import RunAgentInput
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Depends, Request
from copilotkit import LangGraphAGUIAgent

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
        final_answer_content = await soccer_agent.run(
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


# --- Endpoint for Copilotkit Chatbot UI --- 
# This custom endpoint is created by imitate 'from ag_ui_langgraph import add_langgraph_fastapi_endpoint'
def add_custom_copilotkit_endpoint(app: FastAPI, agent: LangGraphAGUIAgent, path: str = "/"):
    from app.api.user import get_current_user
    
    @app.post(path)
    async def langgraph_agent_endpoint(
        input_data: RunAgentInput, 
        request: Request,
        current_user = Depends(get_current_user)
    ):
        accept_header = request.headers.get("accept")
        encoder = EventEncoder(accept=accept_header)
        
        request_agent = agent.clone()
        
        # Add user_id to RunnableConfig to be used in LangGraph nodes
        if request_agent.config is None:
            request_agent.config = {}
        if "configurable" not in request_agent.config:
            request_agent.config["configurable"] = {}
        request_agent.config["configurable"]["user_id"] = current_user.id

        async def event_generator():
            async for event in request_agent.run(input_data):
                yield encoder.encode(event)

        return StreamingResponse(
            event_generator(),
            media_type=encoder.get_content_type()
        )

    @app.get(f"{path}/health")
    def health():
        return {
            "status": "ok",
            "agent": {
                "name": agent.name,
            }
        }
