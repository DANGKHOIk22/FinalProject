import os
import logging
import random
import cv2
import numpy as np
from datetime import datetime
from typing import Type, List, Optional, Literal, Tuple, Union
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langsmith import get_current_run_tree
import dashscope
from dashscope import MultiModalEmbedding
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

    _dashscope_api_key: str = PrivateAttr("")
    _embedding_model: str = PrivateAttr("qwen3-vl-embedding")

    def __init__(self):
        super().__init__()
        os.makedirs(self.output_dir, exist_ok=True)
        self._initialize_dashscope()

    def _initialize_dashscope(self) -> None:
        """Initialize DashScope API for Qwen3-VL-Embedding."""
        self._dashscope_api_key = settings.DASHSCOPE_API_KEY or ""

        if not self._dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is not configured in environment variables.")

        # Test connectivity with a simple request
        try:
            logger.info("✅ DashScope API is configured for Qwen3-VL-Embedding")
        except Exception as exc:
            logger.warning("Error initializing DashScope: %s", exc)

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
        """Preprocess video: extract frames, resize, and save as temporary files.

        :param video_path: Path to the input video file.
        :param desired_fps: Target frames per second to sample from the video.
        :param shortest_edge: The size of the shortest edge after resizing.
        :param jpeg_quality: Quality of JPEG compression (1-100).
        :param max_frames: Maximum number of frames to process.
        :return: A tuple containing a list of original PIL Images and a list of temporary frame file paths.
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
        temp_frame_paths: List[str] = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_original = Image.fromarray(rgb_frame)
                original_frames.append(pil_original)

                # Save resized frame to temporary file
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                rgb_resized = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_resized = Image.fromarray(rgb_resized)

                temp_filename = f"FRAME_TEMP_{timestamp}_{frame_count}.jpg"
                temp_path = os.path.join(self.output_dir, temp_filename)
                pil_resized.save(temp_path, format="JPEG", quality=jpeg_quality, optimize=True)
                temp_frame_paths.append(temp_path)

                if max_frames and len(temp_frame_paths) >= max_frames:
                    break

            frame_count += 1
        logger.info(f"Total frames processed: {len(temp_frame_paths)}")
        cap.release()
        return original_frames, temp_frame_paths

    def _call_qwen_embedding(
        self,
        query: str,
        frame_paths: List[str],
        topk: int = 1,
    ) -> List[int]:
        """Call Qwen3-VL-Embedding via DashScope to select frames based on query.

        :param query: Textual description to match frames against.
        :param frame_paths: List of frame file paths (saved as temporary images).
        :param topk: Number of top frames to return.
        :return: List of selected frame indices (sorted by similarity, descending).
        """
        if not frame_paths:
            return []

        try:
            # Prepare inputs: query as text, each frame as image
            inputs: List[Union[dict, dict]] = [{"text": query}]
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    inputs.append({"image": frame_path})

            if len(inputs) < 2:
                logger.warning("No valid frames to process for embedding")
                return []

            # Call DashScope API
            response = MultiModalEmbedding.call(
                api_key=self._dashscope_api_key,
                model=self._embedding_model,
                input=inputs  # type: ignore
            )

            # Handle API response
            if response.status_code != 200:
                raise ValueError(f"DashScope API error: {response.message}")

            # Extract embeddings
            embeddings_output = response.output.get("embeddings", [])
            if not embeddings_output or len(embeddings_output) < 2:
                logger.error("DashScope returned invalid embeddings")
                raise ValueError("Invalid embeddings from DashScope API")

            # Query embedding is the first one, frame embeddings are the rest
            # Each item in embeddings_output is a dict with "embedding" key
            query_embedding = np.array(embeddings_output[0]["embedding"])
            frame_embeddings = np.array([e["embedding"] for e in embeddings_output[1:]])

            # Calculate cosine similarity
            similarities = []
            for frame_emb in frame_embeddings:
                # Cosine similarity = dot(a, b) / (norm(a) * norm(b))
                similarity = np.dot(query_embedding, frame_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(frame_emb) + 1e-8
                )
                similarities.append(similarity)

            # Sort by similarity descending and get top-k indices
            sorted_indices = np.argsort(similarities)[::-1]
            selected_indices = sorted_indices[:min(topk, len(sorted_indices))].tolist()

            logger.info(f"Selected frames: {selected_indices} with similarities: {[similarities[i] for i in selected_indices]}")
            return selected_indices

        except Exception as e:
            logger.error(f"Error in Qwen embedding call: {e}")
            raise ValueError(f"DashScope embedding call failed: {str(e)}")

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
        """Main execution method using Qwen3-VL-Embedding for frame selection."""
        try:
            logger.info("🔍 Frame selection started for query: %s", query)
            if not material:
                raise ValueError("No video file provided in material.")

            file_path_raw = material[0]
            full_path = os.path.join(self.project_path, file_path_raw) if not os.path.isabs(file_path_raw) else file_path_raw

            # Validate video file exists
            if not os.path.exists(full_path):
                raise ValueError(f"Video file not found at {full_path}")

            logger.info(f"🎞️ Frame Selection from: {full_path}")

            # Preprocess video and extract frames
            original_frames, temp_frame_paths = self._preprocess_video(
                full_path, desired_fps=1, shortest_edge=224, jpeg_quality=100, max_frames=1500
            )

            if not temp_frame_paths:
                logger.warning("No frames extracted from video")
                fallback_path = self._select_random_frame(full_path)
                if fallback_path:
                    return f"No frames extracted. Selected random frame: {fallback_path}", [fallback_path]
                else:
                    return "No frames could be extracted from video", None

            # Call Qwen3-VL-Embedding to select best frames
            try:
                selected_indices = self._call_qwen_embedding(query, temp_frame_paths, topk=1)
            except Exception as qwen_error:
                logger.error(f"Qwen embedding failed: {qwen_error}. Falling back to random frame selection.")
                # Cleanup temp frames before fallback
                for temp_path in temp_frame_paths:
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                del original_frames
                del temp_frame_paths
                fallback_path = self._select_random_frame(full_path)
                if fallback_path:
                    return f"Qwen API failed. Selected a random frame as fallback: {fallback_path}", [fallback_path]
                else:
                    return f"Both Qwen API and random selection failed: {qwen_error}", None

            # Save selected frames and clean up temp files
            if selected_indices and original_frames:
                saved_paths = self._save_selected_frames(selected_indices, original_frames)

                # Clean up temporary frame files
                for temp_path in temp_frame_paths:
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.debug(f"Could not delete temp frame {temp_path}: {e}")

                if saved_paths:
                    saved_paths_str = "\n".join(saved_paths)
                    msg = (
                        f"Successfully selected {len(saved_paths)} frame(s) for query '{query}'. Continue call next tool to analyze extracted frames.\n"
                        f"The most relevant frame is saved at: {saved_paths_str}."
                    )
                    return msg, saved_paths

            # Cleanup if no frames selected
            for temp_path in temp_frame_paths:
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.debug(f"Could not delete temp frame {temp_path}: {e}")

            del original_frames
            del temp_frame_paths

            logger.warning("No frames were selected by Qwen embedding")
            return "No suitable frames found for the query", None

        except Exception as e:
            error_msg = f"Error in frame selection tool: {str(e)}"
            logger.error(error_msg)
            # Send error to LangSmith run tree
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.end(error=error_msg)
            return f"Error during frame selection: {e}", None


        