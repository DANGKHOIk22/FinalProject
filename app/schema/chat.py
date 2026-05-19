from pydantic import BaseModel, Field
from typing import List, Optional
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
    additional_material: Optional[List[str]] = Field(
        None,
        description="Tài liệu bổ sung (ví dụ: đường dẫn file ảnh, video)."
    )