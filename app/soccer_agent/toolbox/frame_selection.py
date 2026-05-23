import os
import re
import logging
from collections import defaultdict
from typing import Type, List, Optional, Literal, Tuple
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langsmith import get_current_run_tree
from dashscope import MultiModalEmbedding
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Range,
    Prefetch, FusionQuery, Fusion, SparseVector,
)
from app.config import settings
from app.config.config import PROJECT_PATH
from langgraph.prebuilt import ToolRuntime
import uuid

logger = logging.getLogger(__name__)


# ==========================================
# 1. Input Schema
# ==========================================

class FrameSelectionInput(BaseModel):
    query: str = Field(description="Description of the desired frame to be selected from the video.")
    video_id: Optional[str] = Field(default=None, description="HLS streaming video ID. Required for frame retrieval from Qdrant.")
    current_time: Optional[float] = Field(default=None, description="Current video playback position in seconds.")
    intent: Literal["current", "recent", "specific", "none"] = Field(
        default="none",
        description=(
            "Temporal intent extracted from the query: "
            "current=last 10s from current_time, "
            "recent=last 60s from current_time, "
            "specific=extract exact time or range from query text, "
            "none=no temporal filter."
        ),
    )
    time_start: Optional[float] = Field(default=None, description="Start of temporal filter in seconds from video start. Used when intent=specific.")
    time_end: Optional[float] = Field(default=None, description="End of temporal filter in seconds from video start. Used when intent=specific.")


# ==========================================
# 2. Tool
# ==========================================

class FrameSelectionTool(BaseTool):
    name: str = "frame_selection"
    description: str = (
        "Selects the most relevant frame(s) for a given query from an HLS video. "
        "Supports temporal filtering (current=last 10s, recent=last 60s, specific=time extracted from query). "
        "Returns frame file paths appended to additional_material for downstream visual analysis tools."
    )
    args_schema: Type[BaseModel] = FrameSelectionInput  # type: ignore
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    project_path: str = PROJECT_PATH
    output_dir: str = os.path.join(PROJECT_PATH, "temporary", "frames")

    _dashscope_api_key: str = PrivateAttr("")
    _embedding_model: str = PrivateAttr("tongyi-embedding-vision-flash")
    _bm25: SparseTextEmbedding = PrivateAttr()

    def __init__(self):
        super().__init__()
        os.makedirs(self.output_dir, exist_ok=True)
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
        return FieldCondition(key="timestamp", range=ts_range)

    # ── Qdrant two-call query + Python score merge ────────────────────────────

    def _query_qdrant(self, inp: FrameSelectionInput, top_k: int = 5) -> Tuple[str, Optional[List[str]]]:
        if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
            return "Qdrant is not configured.", None

        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

        query_vec = self._embed_text(inp.query)
        sparse_vec = self._embed_sparse(inp.query)
        timestamp_condition = self._build_timestamp_filter(inp)

        def chunk_filter(chunk_type: str) -> Filter:
            conditions = [
                FieldCondition(key="video_id", match=MatchValue(value=inp.video_id)),
                FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)),
            ]
            if timestamp_condition:
                conditions.append(timestamp_condition)
            return Filter(must=conditions)

        # Call 1: cross-modal image search (text query → vision multi-vectors, MAX_SIM)
        image_hits = client.query_points(
            collection_name=settings.QDRANT_HLS_COLLECTION_NAME,
            query=query_vec,
            using="dense_image",
            query_filter=chunk_filter("image"),
            limit=top_k,
            with_payload=True,
        ).points

        # Call 2: transcript search — RRF over dense_text + BM25 sparse
        transcript_hits = client.query_points(
            collection_name=settings.QDRANT_HLS_COLLECTION_NAME,
            prefetch=[
                Prefetch(query=query_vec,  using="dense_text", filter=chunk_filter("transcript"), limit=top_k),
                Prefetch(query=sparse_vec, using="sparse",     filter=chunk_filter("transcript"), limit=top_k),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        ).points

        # Merge scores by segment_index; only keep segments that have an image point
        scores: dict = defaultdict(float)
        image_payloads: dict = {}

        for hit in image_hits:
            seg = (hit.payload or {}).get("segment_index")
            if seg is not None:
                scores[seg] += hit.score
                image_payloads[seg] = (hit.payload or {}).get("frame_paths", [])

        for hit in transcript_hits:
            seg = (hit.payload or {}).get("segment_index")
            if seg is not None:
                scores[seg] += hit.score

        top_segments = sorted(
            (seg for seg in scores if seg in image_payloads),
            key=lambda s: scores[s],
            reverse=True,
        )[:top_k]

        frame_paths = [p for seg in top_segments for p in image_payloads[seg]]
        logger.info(f"Frame selection found {frame_paths} for query '{inp.query}' with intent '{inp.intent}' in video '{inp.video_id}'")

        if not frame_paths:
            return f"No frames found for video '{inp.video_id}'.", None

        paths_str = "\n".join(f"  {p}" for p in frame_paths)
        msg = (
            f"Retrieved {len(frame_paths)} frame(s) from video '{inp.video_id}'.\n"
            f"Frames:\n{paths_str}\n"
            "Continue to call next tool to analyze the extracted frames."
        )
        return msg, frame_paths

    # ── Entry point ───────────────────────────────────────────────────────────

    def _run(
        self,
        query: str,
        video_id: Optional[str] = None,
        current_time: Optional[float] = None,
        intent: str = "none",
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Optional[List[str]]]:
        try:
            logger.info("Frame selection — query: %s, intent: %s, video: %s", query, intent, video_id)

            if not video_id:
                raise ValueError("video_id is required.")

            inp = FrameSelectionInput(
                query=query,
                video_id=video_id,
                current_time=current_time,
                intent=intent,  # type: ignore
                time_start=time_start,
                time_end=time_end,
            )
            return self._query_qdrant(inp)

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
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.end(error=error_msg)
            return error_msg, None
