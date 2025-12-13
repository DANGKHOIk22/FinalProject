import os
import logging
import random
import numpy as np
import cv2
import ffmpeg
from datetime import datetime
from typing import Type, List, Optional, Literal, Tuple
from PIL import Image

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from app.config.settings import Settings
from app.config.config import PROJECT_PATH

# Giả sử bạn để class CLIPHelper ở đường dẫn này
from app.toolbox.helper.clip import CLIPHelper 

logger = logging.getLogger(__name__)

# ==========================================
# 1. Input Schema
# ==========================================

class FrameSelectionInput(BaseModel):
    query: str = Field(description="Mô tả chi tiết về khung hình (frame) bạn muốn tìm trong video.")
    material: List[str] = Field(description="Danh sách đường dẫn file video. Thường chỉ chứa 1 phần tử.")
    
class FrameSelectionTool(BaseTool):
    name: str = "frame_selection"
    description: str = """
    Given a description query and a video of soccer game, the tool selects the frame that best matches the prompt 
    and saves that frame as an image. This image is crucial for subsequent visual analysis steps.
    Input: A text prompt describing the scene and a video file path.
    Output: The file path of the saved image frame.
    """
    args_schema: Type[BaseModel] = FrameSelectionInput
    
    # Trả về cả Content (Text) và Artifact (File Path)
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    
    project_path: str = PROJECT_PATH
    output_dir: str = os.path.join(PROJECT_PATH, "log", "frames") # Folder lưu ảnh output
    
    # Instance của CLIPHelper (Singleton hoặc khởi tạo 1 lần)
    clip_helper: Optional[CLIPHelper] = None

    def __init__(self):
        super().__init__()
        # Tạo folder output nếu chưa có
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Khởi tạo CLIPHelper
        try:
            self.clip_helper = CLIPHelper()
            logger.info("CLIPHelper initialized successfully for Frame Selection.")
        except Exception as e:
            logger.error(f"Failed to initialize CLIPHelper: {e}")

    def _select_random_frame(self, video_path: str) -> Optional[str]:
        """Chọn ngẫu nhiên 1 frame nếu CLIP thất bại."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video for random selection: {video_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return None
            
            random_frame_idx = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"FRAME_RANDOM_{timestamp}.jpg"
            output_path = os.path.join(self.output_dir, output_filename)
            
            cv2.imwrite(output_path, frame)
            logger.info(f"Random frame saved at: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in random frame selection: {e}")
            return None

    def _process_video_with_clip(self, video_path: str, query: str) -> Optional[str]:
        """Dùng ffmpeg + CLIP để tìm frame khớp nhất."""
        if not self.clip_helper:
            logger.error("CLIP Helper is not available.")
            return None

        best_similarity = -np.inf
        best_frame = None
        
        try:
            # Lấy thông tin video
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if not video_stream:
                return None
                
            width = int(video_stream['width'])
            height = int(video_stream['height'])

            # Chạy ffmpeg pipe để lấy frame mỗi giây (r=1)
            process = (
                ffmpeg.input(video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=1)
                .run_async(pipe_stdout=True, quiet=True)
            )

            logger.info(f"Scanning video for query: '{query}'...")
            
            while True:
                # Đọc raw bytes từ pipe
                in_bytes = process.stdout.read(width * height * 3)
                if not in_bytes:
                    break
                
                # Convert bytes sang PIL Image
                frame = Image.frombytes('RGB', (width, height), in_bytes)
                
                # Tính similarity bằng CLIPHelper
                # Hàm get_unit_per_image trả về float score
                similarity = self.clip_helper.get_unit_per_image(images=frame, texts=[query])
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_frame = frame.copy() # Copy để giữ lại frame tốt nhất
            
            process.wait()

            if best_frame:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"FRAME_SELECTED_{timestamp}.jpg"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Save PIL Image
                best_frame.save(output_path, quality=95, subsampling=0)
                logger.info(f"Best frame found (Score: {best_similarity:.4f}) saved at: {output_path}")
                return output_path
            
            return None

        except Exception as e:
            logger.error(f"Error processing video with CLIP: {e}")
            return None

    def _run(self, query: str, material: List[str], run_manager: Optional[CallbackManagerForToolRun] = None) -> Tuple[str, Optional[str]]:
        """Hàm thực thi chính."""
        if not material:
            return "Error: No video material provided.", None

        file_path_raw = material[0]
        full_path = os.path.join(self.project_path, file_path_raw)
        
        # Fallback check path
        if not os.path.exists(full_path):
            if os.path.exists(file_path_raw):
                full_path = file_path_raw
            else:
                return f"Error: Video file not found at {full_path}", None

        logger.info(f"🎞️ Frame Selection from: {full_path}")

        # 1. Thử chọn bằng CLIP (AI)
        selected_path = self._process_video_with_clip(full_path, query)

        if selected_path:
            msg = f"Successfully selected the best matching frame for query '{query}'."
            return msg, selected_path

        # 2. Fallback: Chọn Random nếu CLIP thất bại hoặc không tìm thấy frame
        logger.warning("CLIP selection failed or yielded no result. Falling back to Random Selection.")
        random_path = self._select_random_frame(full_path)
        
        if random_path:
            msg = "Could not find an exact match. A random frame has been selected as a fallback."
            return msg, random_path
        
        return "Failed to extract any frame from the video.", full_path
