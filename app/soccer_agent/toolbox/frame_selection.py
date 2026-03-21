import os
import io
import base64
import logging
import random
import cv2
from datetime import datetime
from typing import Type, List, Optional, Literal, Tuple
import requests
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langsmith import get_current_run_tree
from app.config import settings
from app.config.config import PROJECT_PATH

logger = logging.getLogger(__name__)

# ==========================================
# 1. Input Schema
# ==========================================

class FrameSelectionInput(BaseModel):
    query: str = Field(description="Description of the desired frame to be selected from the video. This should be a concise text prompt. For example, 'A soccer player scoring a goal' or 'A goalkeeper making a save'.")
    material: List[str] = Field(description="List of video file paths. Usually contains only one element.")
    
class FrameSelectionTool(BaseTool):
    name: str = "frame_selection"
    description: str = """
    Given a description query and a video of soccer game, the tool selects the frame that best matches the prompt 
    and saves that frame as an image. This image is crucial for subsequent visual analysis steps.
    Input: A text prompt describing the scene and a video file path.
    Output: The file path of the saved image frame.
    """
    args_schema: Type[BaseModel] = FrameSelectionInput #type: ignore
    
    # Trả về cả Content (Text) và Artifact (File Path)
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    project_path: str = PROJECT_PATH
    output_dir: str = os.path.join(PROJECT_PATH, "temporary", "frames")

    _cg_endpoint_uri: str = PrivateAttr("")
    _cg_endpoint_key: str = PrivateAttr("")
    _cg_payload_header: dict = PrivateAttr({})

    def __init__(self):
        super().__init__()
        os.makedirs(self.output_dir, exist_ok=True)
        self._initialize_endpoint()

    def _initialize_endpoint(self) -> None:
        # Prefer combined CLIP+GroundingDINO endpoint; fallback to legacy CLIP endpoint if needed
        self._cg_endpoint_uri = (
            settings.CLIP_GROUNDINGDINO_ENDPOINT_URI
            or settings.CLIP_ENDPOINT_URI
            or ""
        )
        self._cg_endpoint_key = (
            settings.CLIP_GROUNDINGDINO_ENDPOINT_KEY
            or settings.CLIP_ENDPOINT_KEY
            or ""
        )

        if not self._cg_endpoint_uri or not self._cg_endpoint_key:
            raise ValueError("CLIP/GroundingDINO endpoint URI/Key is not configured.")

        self._cg_payload_header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cg_endpoint_key}",
        }

        try:
            response = requests.post(
                url=self._cg_endpoint_uri,
                headers=self._cg_payload_header,
                json={"ping": True},
                timeout=60,
            )
            if response.status_code == 200:
                logger.info("✅ CLIP/GroundingDINO endpoint is reachable")
            else:
                logger.warning(
                    "CLIP/GroundingDINO endpoint healthcheck returned %s: %s",
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.warning("Could not reach CLIP/GroundingDINO endpoint during init: %s", exc)

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

    def _preprocess_video(
        self,
        video_path: str,
        desired_fps: int = 1,
        shortest_edge: int = 224,
        jpeg_quality: int = 100,
        max_frames: int = 1500,
    ) -> Tuple[List[Image.Image], List[str]]:
        """Preprocess video: extract frames, resize, compress, and encode to base64. This function helps reduce the payload size for CLIP endpoint.

        :param video_path: Path to the input video file.
        :param desired_fps: Target frames per second to sample from the video. 
        :param shortest_edge: The size of the shortest edge after resizing. 
        :param jpeg_quality: Quality of JPEG compression (1-100).
        :param max_frames: Maximum number of frames to process.
        :return: A tuple containing a list of original PIL Images and a list of base64-encoded JPEG strings.
        """

        # Open video file with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Calculate new dimensions while maintaining aspect ratio
        min_dimension = min(width, height) if (width and height) else shortest_edge
        scale_factor = shortest_edge / min_dimension
        new_width = int(width * scale_factor) if width else shortest_edge
        new_height = int(height * scale_factor) if height else shortest_edge
        new_width = new_width if new_width % 2 == 0 else new_width - 1
        new_height = new_height if new_height % 2 == 0 else new_height - 1
        if new_width <= 0 or new_height <= 0:
            new_width = new_height = max(2, shortest_edge)

        frame_skip = max(1, int(round(fps / desired_fps))) if fps and desired_fps > 0 else 1
        logger.info(f"Frame skip calculated: {frame_skip} (fps: {fps}, desired_fps: {desired_fps})")

        # Extract and process frames
        frame_count = 0
        original_frames: List[Image.Image] = []
        processed_b64: List[str] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_original = Image.fromarray(rgb_frame)
                original_frames.append(pil_original)

                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                rgb_resized = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_resized = Image.fromarray(rgb_resized)

                buffer = io.BytesIO()
                pil_resized.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                processed_b64.append(encoded)

                if max_frames and len(processed_b64) >= max_frames:
                    break

            frame_count += 1
        logger.info(f"Total frames processed: {len(processed_b64)}")
        cap.release()
        return original_frames, processed_b64

    def _call_clip_endpoint(
        self,
        query: str,
        frames_b64: List[str],
        topk: int = 1,
        threshold: float = 0.0,
    ) -> List[int]:
        """Call CLIP endpoint to select frames based on query.
         :param query: Textual description to match frames against.
         :param frames_b64: List of base64-encoded JPEG frames.
         :param topk: Number of top frames to return.
         :param threshold: Similarity threshold for frame selection.
         :return: List of selected frame indices.
        """
        if not frames_b64:
            return []

        # Prepare payload for CLIP endpoint
        payload = {
            "task": "clip",
            "query": query,
            "images": frames_b64,
            "topk": max(1, min(topk, 10)),
            "threshold": max(0.0, min(threshold, 1.0)),
        }

        # Make request to CLIP endpoint
        response = requests.post(
            url=self._cg_endpoint_uri,
            headers=self._cg_payload_header,
            json=payload,
            timeout=120,
        )

        # Parse response
        try:
            response_dict = response.json()
        except Exception:
            raise ValueError("Invalid JSON response from CLIP endpoint. Trying to retry the request later.")
            

        # Handle errors
        if response.status_code != 200 or not response_dict.get("success", False):
            logger.error(
                "CLIP/GroundingDINO endpoint request failed (%s): %s",
                response.status_code,
                response_dict.get("error") or response.text,
            )
            raise ValueError("CLIP endpoint request failed. Please check the logs for details.")

        # Extract selected frame indices
        selected_indices = response_dict.get("selected_frames") or []
        if not isinstance(selected_indices, list):
            logger.error("CLIP endpoint returned invalid selected_frames format")
            raise ValueError("Invalid selected_frames format from CLIP endpoint.")


        return [
            int(idx)
            for idx in selected_indices
            if isinstance(idx, (int, float)) and 0 <= int(idx) < len(frames_b64)
        ]

    def _save_selected_frames(self, indices: List[int], frames: List[Image.Image]) -> List[str]:
        """Save selected frames to disk and return their paths.
        :param indices: List of frame indices to save.
        :param frames: List of original PIL Image frames.
        :return: List of file paths where frames are saved.
        """
        saved_paths: List[str] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx in indices:
            if idx < 0 or idx >= len(frames):
                continue
            output_filename = f"FRAME_SELECTED_{timestamp}_{idx}.jpg"
            output_path = os.path.join(self.output_dir, output_filename)
            frames[idx].save(output_path, format="JPEG", quality=100, subsampling=0)
            saved_paths.append(output_path)

        return saved_paths

    def _run(self, query: str, material: List[str], run_manager: Optional[CallbackManagerForToolRun] = None) -> Tuple[str, Optional[List[str]]]:
        """Hàm thực thi chính."""
        try:
            logger.info("🔍 Frame selection started for query: %s", query)
            if not material:
                raise ValueError("No video file provided in material.")

            file_path_raw = material[0]
            full_path = os.path.join(self.project_path, file_path_raw) if not os.path.isabs(file_path_raw) else file_path_raw
        
            # Fallback check path
            if not os.path.exists(full_path):
                raise ValueError(f"Video file not found at {full_path}")

            logger.info(f"🎞️ Frame Selection from: {full_path}")
            # Preprocess video and call CLIP endpoint
            original_frames, processed_b64 = self._preprocess_video(full_path, desired_fps=1, shortest_edge=224, jpeg_quality=100, max_frames=1500)
            selected_indices = self._call_clip_endpoint(query, processed_b64, topk=1, threshold=0.0)

            # Process and save selected frames
            if selected_indices and original_frames:
                saved_paths = self._save_selected_frames(selected_indices, original_frames)
                if saved_paths:
                    saved_paths_str = "\n".join(saved_paths)
                    msg = (
                        f"Successfully selected {len(saved_paths)} frame(s) for query '{query}'. Continue call next tool to analyze extracted frames.\n"
                        f"The most relevant frame is saved at: "
                        f"{saved_paths_str}."
                    )
                    return msg, saved_paths
            del original_frames
            del processed_b64
        except Exception as e:
            error_msg = f"Error in frame selection tool: {str(e)}"
            logger.error(error_msg)                       
            # Send error to LangSmith run tree
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.end(
                    error=error_msg
                )
            return f"Error during frame selection: {e}", None


        