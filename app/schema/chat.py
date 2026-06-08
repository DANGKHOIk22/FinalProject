from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
class ChatRequest(BaseModel):
    """Schema cho input của người dùng."""
    user_id: str = Field(
        ..., description="ID duy nhất của người dùng.")
    session_id: Optional[str] = Field(
        None, description="ID của phiên trò chuyện. Nếu không có, mỗi request tạo session mới.")
    user_query: str = Field(
        ...,
        description="Câu hỏi của người dùng về bóng đá."
    )
    additional_material: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Tài liệu bổ sung. Dict shape: {'game_id': str | None, 'image_id': List[str]}. "
            "game_id: ID returned by POST /hls/sessions, required when the user is watching a "
            "live-simulated stream and the agent should query indexed frames. "
            "image_id: list of media file paths (e.g. image/video frame paths)."
        )
    )