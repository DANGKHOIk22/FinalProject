"""
Incremental Segment Indexer
===========================
Embeds video frames and transcripts per-segment using DashScope text embeddings
and persists them to Qdrant, keyed by video_id, for later agent-side retrieval.

Schema: one point per selected keyframe. Each point carries:
  - dense_caption → DashScope text-embedding-v4 embedding of VLM-generated frame caption
  - dense_text    → DashScope text-embedding-v4 embedding of the segment transcript
  - sparse        → FastEmbed BM25 sparse embedding of caption + transcript
  payload: video_id, frame_path (str), segment_index, timestamp (segment start),
           transcript, caption
"""

import json
import os
import re
import uuid
import base64
import logging
from http import HTTPStatus
from typing import Dict, List, Optional

import dashscope
from fastembed import SparseTextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


logger = logging.getLogger(__name__)

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

_CAPTION_SYSTEM_PROMPT = """You are a helpful assistant for describing people in sports images.

For each image:
- Describe the main person or persons briefly
- Mention clothing/uniform
- Mention actions or poses
- Mention visible appearance if relevant
- Mention possible role (player, coach, referee, etc.)

Keep descriptions short (one sentence per person).
Do not invent details that are not visible.

Return ONLY a JSON array in this format:

[
    {
        "image_index": 0,
        "description": "A player in a blue jersey is running with the ball."
    }
]

No markdown. No explanations."""

_TEXT_EMBEDDING_MODEL = "text-embedding-v4"
_CAPTION_MODEL = "qwen3-vl-flash-2026-01-22"
_CAPTION_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class IncrementalSegmentIndexer:
    """
    Generates VLM captions for keyframes, embeds captions and transcripts with
    text-embedding-v4, and persists one Qdrant point per keyframe (flat per-frame schema).

    Usage::

        indexer = IncrementalSegmentIndexer(dashscope_api_key=api_key,
                                          qdrant_client=client,
                                          qdrant_collection="hls_frame_index",
                                          video_id="abc123")
        indexer.index_segment_frames(
            frame_paths=["f0.jpg", "f1.jpg"],
            transcript="commentary text",
            segment_index=0,
            timestamp=0.0,
        )
    """

    def __init__(
        self,
        dashscope_api_key: str,
        qdrant_client: Optional[QdrantClient] = None,
        qdrant_collection: str = "hls_frame_index",
        video_id: Optional[str] = None,
    ):
        self._api_key = dashscope_api_key
        self._qdrant = qdrant_client
        self._collection = qdrant_collection
        self._video_id = video_id
        self._collection_ready = False

        self._all_frame_paths: List[str] = []
        self._prev_transcript: str = ""
        self._bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")

        if dashscope_api_key:
            self._openai_client = OpenAI(
                api_key=dashscope_api_key,
                base_url=_CAPTION_BASE_URL,
            )
        else:
            self._openai_client = None

    @property
    def frame_paths(self) -> List[str]:
        return self._all_frame_paths

    @property
    def total_indexed(self) -> int:
        return len(self._all_frame_paths)

    # ── Public: frame indexing ────────────────────────────────────────────────

    def index_segment_frames(
        self,
        frame_paths: List[str],
        transcript: str,
        segment_index: int,
        timestamp: float,
    ) -> None:
        """
        Generate captions for each frame, embed captions and transcript with
        text-embedding-v4, then upsert one Qdrant point per frame.

        :param frame_paths: Selected keyframe file paths for this segment.
        :param transcript: Segment-level STT transcript (may be empty string).
        :param segment_index: Zero-based segment index within the session.
        :param timestamp: Segment start time in seconds (used for all frame points).
        """
        if not frame_paths:
            return

        self._all_frame_paths.extend(frame_paths)

        if not self._api_key:
            return

        # Build overlapped transcript for BM25 cross-boundary context
        tail = self._prev_transcript[-100:] if self._prev_transcript else ""
        overlapped_transcript = tail + transcript
        self._prev_transcript = transcript
        cleaned_transcript = self._preprocess_text(overlapped_transcript) if overlapped_transcript else ""

        # Generate captions for all frames in one batch call
        captions = self._batch_caption_frames(frame_paths)

        # Embed transcript once — shared across all frame points for this segment
        try:
            text_emb = self._compute_text_dense_embedding(cleaned_transcript) if cleaned_transcript else None
        except Exception as e:
            logger.warning(f"Transcript embedding failed for segment {segment_index}: {e}")
            text_emb = None

        try:
            bm25_transcript = list(self._bm25.embed([cleaned_transcript]))[0] if cleaned_transcript else None
        except Exception as e:
            logger.warning(f"Transcript BM25 failed for segment {segment_index}: {e}")
            bm25_transcript = None

        if not (self._qdrant and self._video_id):
            return

        for i, frame_path in enumerate(frame_paths):
            if i not in captions:
                logger.warning(f"No caption for frame index {i} (segment {segment_index}), skipping.")
                continue

            caption = captions[i]
            cleaned_caption = self._preprocess_text(caption)
            bm25_input = (cleaned_caption + " " + cleaned_transcript).strip() if cleaned_transcript else cleaned_caption

            try:
                caption_emb = self._compute_text_dense_embedding(caption)
            except Exception as e:
                logger.warning(f"Caption embedding failed for {frame_path}: {e}")
                continue

            self._ensure_collection(len(caption_emb))

            vector: dict = {"dense_caption": caption_emb}
            if text_emb is not None:
                vector["dense_text"] = text_emb

            try:
                sparse_result = list(self._bm25.embed([bm25_input]))[0]
                vector["sparse"] = SparseVector(
                    indices=sparse_result.indices.tolist(),
                    values=sparse_result.values.tolist(),
                )
            except Exception as e:
                logger.warning(f"BM25 embedding failed for frame {frame_path}: {e}")

            self._qdrant.upsert(
                collection_name=self._collection,
                points=[PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "video_id": self._video_id,
                        "frame_path": frame_path,
                        "segment_index": segment_index,
                        "timestamp": timestamp,
                        "transcript": transcript,
                        "caption": caption,
                    },
                )],
            )
            logger.debug(f"Persisted frame point: {frame_path} (segment={segment_index})")

    # ── Internal: captioning ──────────────────────────────────────────────────

    def _batch_caption_frames(self, frame_paths: List[str]) -> Dict[int, str]:
        """
        Call qwen3-vl-flash to generate one caption per frame in a single batch request.

        Returns a dict mapping frame index → caption string.
        Returns {} on any failure so the caller can skip all frames gracefully.
        """
        if not self._openai_client or not frame_paths:
            return {}

        content = []
        for frame_path in frame_paths:
            if not os.path.exists(frame_path):
                logger.warning(f"Frame file not found: {frame_path}")
                continue
            try:
                with open(frame_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            except Exception as e:
                logger.warning(f"Failed to read frame {frame_path}: {e}")

        if not content:
            return {}

        n = len(content)
        content.append({
            "type": "text",
            "text": f"Describe persons in these {n} sports images from image index 0 to {n - 1}.",
        })

        try:
            response = self._openai_client.chat.completions.create(
                model=_CAPTION_MODEL,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": _CAPTION_SYSTEM_PROMPT}]},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
                top_p=0.1,
            )
            raw = response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Caption API call failed: {e}")
            return {}

        try:
            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            parsed = json.loads(text)
            return {item["image_index"]: item["description"] for item in parsed}
        except Exception as e:
            logger.warning(f"Caption JSON parse failed: {e}. Raw: {raw[:200]}")
            return {}

    # ── Internal: embeddings ──────────────────────────────────────────────────

    def _compute_text_dense_embedding(self, text: str) -> List[float]:
        resp = dashscope.TextEmbedding.call(
            model=_TEXT_EMBEDDING_MODEL,
            input=text,
            api_key=self._api_key,
        )
        if resp.status_code != HTTPStatus.OK:
            raise ValueError(f"DashScope TextEmbedding error: {resp.message}")

        raw = resp.output.get("embeddings", [])
        if not raw:
            raise ValueError("Invalid text embedding response")

        return raw[0]["embedding"]

    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    # ── Internal: Qdrant ──────────────────────────────────────────────────────

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if self._collection not in existing:
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config={
                    "dense_caption": VectorParams(size=vector_size, distance=Distance.COSINE),
                    "dense_text": VectorParams(size=vector_size, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
            )
            self._qdrant.create_payload_index(
                collection_name=self._collection,
                field_name="video_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self._qdrant.create_payload_index(
                collection_name=self._collection,
                field_name="segment_index",
                field_schema=PayloadSchemaType.INTEGER,
            )
            logger.info(f"Created Qdrant collection '{self._collection}' (dim={vector_size})")
        self._collection_ready = True
