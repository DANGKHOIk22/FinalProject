import os
import logging
import base64
import io
import cv2
import numpy as np
from typing import Type, List, Optional, Literal, Tuple
from PIL import Image

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import CallbackManagerForToolRun

from app.config.settings import Settings
from app.config.config import PROJECT_PATH, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ==========================================
# 1. Input Schema
# ==========================================

class ScoreTimeInput(BaseModel):
    query: str = Field(description="Câu hỏi cụ thể về tỉ số hoặc thời gian (VD: 'What is the current score?').")
    material: List[str] = Field(description="Danh sách đường dẫn file (ảnh hoặc video). Thường chỉ chứa 1 phần tử.")

class MatchVisualData(BaseModel):
    """Thông tin thô được trích xuất trực tiếp từ ảnh (Info)."""
    teams: str = Field(description="Tên các đội bóng nhận diện được trên bảng điểm (VD: 'Arsenal vs Liverpool'). Nếu không rõ để 'unknown'.")
    score: str = Field(description="Tỉ số hiện tại hiển thị trên màn hình (VD: '2-1').")
    game_time: str = Field(description="Thời gian thi đấu hiển thị trên đồng hồ (VD: '34:12').")

class VLMResponse(BaseModel):
    """Cấu trúc trả về tổng thể."""
    llm_response: str = Field(description="Câu trả lời tự nhiên, chi tiết cho câu hỏi của người dùng (VD: 'Tỉ số hiện tại là 2-1 nghiêng về Arsenal...').")
    info: MatchVisualData = Field(description="Dữ liệu cứng trích xuất từ ảnh, không bao gồm lời giải thích.")

# ==========================================
# 2. Tool Definition
# ==========================================

class ScoreTimeRecognitionTool(BaseTool):
    name: str = "score_time_recognition"
    description: str = """
    Recognizes the score and time of the game from a soccer broadcast video clip or image using Vision Language Model.
    Input: A query about score/time and a list containing a file path to an image or video.
    Output: The textual answer containing the score and game time visible in the footage.
    """
    args_schema: Type[BaseModel] = ScoreTimeInput
    
    # Tool này trả về cả content (text) và artifact (file path)
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    
    project_path: str = PROJECT_PATH
    llm: ChatGoogleGenerativeAI = None

    def __init__(self):
        super().__init__()
        self.llm = ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL,
            temperature=0,
            google_api_key=Settings.GOOGLE_API_KEY
        )

    def _extract_frame_from_video(self, video_path: str) -> Optional[Image.Image]:
        """Trích xuất frame giữa của video và chuyển thành PIL Image."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video file: {video_path}")
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return None

            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.error("Failed to read frame from video")
                return None

            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return None

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to Base64 string for Gemini."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _run(self, query: str, material: List[str], run_manager: Optional[CallbackManagerForToolRun] = None) -> Tuple[str, Optional[str]]:
        """
        Hàm thực thi chính.
        Trả về: (content_str, artifact_path)
        """
        if not material:
            return "Error: No material provided.", None

        file_path_raw = material[0]
        # Xử lý đường dẫn tuyệt đối/tương đối
        full_path = os.path.join(self.project_path, file_path_raw)
        
        # Fallback check path
        if not os.path.exists(full_path):
            if os.path.exists(file_path_raw):
                full_path = file_path_raw
            else:
                return f"Error: File not found at {full_path}", None

        logger.info(f"🎥 VLM Analyzing Score/Time from: {full_path}")

        # 1. Load Image (từ file ảnh hoặc trích xuất từ video)
        pil_image = None
        if full_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            pil_image = self._extract_frame_from_video(full_path)
        elif full_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                pil_image = Image.open(full_path)
            except Exception as e:
                return f"Error opening image: {e}", full_path
        else:
            return "Error: Unsupported file format.", full_path

        if pil_image is None:
            return "Error: Could not process image/video material.", full_path

        # 2. Chạy Gemini VLM (Vision Language Model)
        try:
            # Chuẩn bị ảnh base64
            img_base64 = self._image_to_base64(pil_image)
            
            # Prompt tối ưu cho VLM
            final_prompt = f"""
            Analyze this soccer broadcast frame carefully.
            The user asks: "{query}"
            
            Task:
            1. Locate the scoreboard overlay (usually top-left, top-right, or bottom).
            2. Identify the Team Names (often 3-letter abbreviations or logos).
            3. Read the Current Score.
            4. Read the Game Time (Minutes:Seconds).
            
            Return the answer in a clear, concise sentence containing the identified teams, score, and time.
            """

            # Tạo message đa phương tiện (Multimodal)
            message = HumanMessage(
                content=[
                    {"type": "text", "text": final_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
            structured_llm = self.llm.with_structured_output(VLMResponse)

            response: VLMResponse = structured_llm.invoke([message])
            
            logger.info(f"✅ VLM Result: {response.llm_response}")
            logger.debug(f"🔍 Extracted Info: {response.info}")
            
            # Trả về Content cho LLM đọc và Artifact là đường dẫn file gốc
            return response.llm_response, response.info

        except Exception as e:
            error_msg = f"Error analyzing image with VLM: {str(e)}"
            logger.error(error_msg)
            return error_msg, full_path