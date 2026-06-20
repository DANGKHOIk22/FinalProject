import uuid
import logging
import asyncio
from typing import Any, Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.api.deps import get_current_user
from app.database.models import User
from app.soccer_agent.memory.chat_history import ConversationHistoryManager

router = APIRouter()
logger = logging.getLogger(__name__)

class HistoryMessage(BaseModel):
    id: str
    role: str
    content: Any

def _serialize_content(content) -> Any:
    """Convert LangChain message content to JSON-serializable format for AG-UI."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks = []
        for block in content:
            if hasattr(block, 'model_dump'):
                block = block.model_dump()
            
            if not isinstance(block, dict):
                continue
                
            block_type = block.get("type")
            if block_type == "text":
                blocks.append({
                    "type": "text",
                    "text": block.get("text", "")
                })

            # TODO: Implement the logic to revoke the image/video url from mediaregistry
            # elif block_type == "image_url":
            #     img_url = block.get("image_url", {}).get("url", "")
            #     blocks.append({
            #         "type": "image",
            #         "source": {
            #             "type": "url",
            #             "value": img_url
            #         }
            #     })
        
        # Optimize: if it's purely text blocks, join them into a string
        if all(b.get("type") == "text" for b in blocks):
            return "\n".join(b.get("text", "") for b in blocks)
            
        return blocks if blocks else ""
    return str(content)

@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: Annotated[str, Path(title="Thread ID")],
    current_user: Annotated[User, Depends(get_current_user)]
) -> list[HistoryMessage]:
    """
    Load conversation history from PostgresChatMessageHistory for a specific thread.
    Only returns HumanMessage and AIMessage (ignores ToolMessage and AIMessage with tool_calls).
    """
    try:
        manager = ConversationHistoryManager(session_id=thread_id)
        messages = await asyncio.to_thread(manager.load_messages)
    except Exception as e:
        logger.error(f"Failed to load messages for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load conversation history")

    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append(HistoryMessage(
                id=msg.id or str(uuid.uuid4()),
                role="user",
                content=_serialize_content(msg.content)
            ))
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            result.append(HistoryMessage(
                id=msg.id or str(uuid.uuid4()),
                role="assistant",
                content=_serialize_content(msg.content)
            ))

    return result
