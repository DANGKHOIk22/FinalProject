"""
Incremental Segment Indexer
===========================
Embeds video frames and transcripts per-segment using DashScope vision embeddings
and persists them to Qdrant, keyed by video_id, for later agent-side retrieval.

Schema: one point per selected keyframe. Each point carries:
  - dense_image  → single DashScope vision embedding for the frame
  - dense_text   → DashScope text embedding of the segment transcript
  - sparse       → FastEmbed BM25 sparse embedding of the segment transcript
  payload: video_id, frame_path (str), segment_index, timestamp (segment start), transcript
"""

import os
import re
import uuid
import base64
import logging
from typing import List, Optional

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
from fastembed import SparseTextEmbedding

import dashscope


logger = logging.getLogger(__name__)


class IncrementalSegmentIndexer:
    """
    Embeds selected keyframes alongside their segment transcript and persists
    one Qdrant point per keyframe (flat per-frame schema).

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
        embedding_model: str = "tongyi-embedding-vision-flash",
        batch_size: int = 50,
        qdrant_client: Optional[QdrantClient] = None,
        qdrant_collection: str = "hls_frame_index",
        video_id: Optional[str] = None,
    ):
        self._api_key = dashscope_api_key
        self._model = embedding_model
        self._batch_size = batch_size

        self._qdrant = qdrant_client
        self._collection = qdrant_collection
        self._video_id = video_id
        self._collection_ready = False

        self._all_frame_paths: List[str] = []
        self._bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")

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
        Embed each frame and the shared transcript, then upsert one Qdrant point
        per frame (flat per-frame schema).

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

        # Compute transcript embeddings once — shared across all k frame points
        cleaned = self._preprocess_transcript(transcript) if transcript else ""
        try:
            text_emb = self._compute_text_dense_embedding(cleaned) if cleaned else None
            sparse_result = list(self._bm25.embed([cleaned]))[0] if cleaned else None
        except Exception as e:
            logger.warning(f"Transcript embedding failed for segment {segment_index}: {e}")
            text_emb = None
            sparse_result = None

        if not (self._qdrant and self._video_id):
            return

        # Upsert one point per frame
        for frame_path in frame_paths:
            try:
                image_embs = self._compute_image_embeddings([frame_path])
                if not image_embs:
                    continue
                image_emb = image_embs[0]
            except Exception as e:
                logger.warning(f"Image embedding failed for {frame_path}: {e}")
                continue

            self._ensure_collection(len(image_emb))

            vector: dict = {"dense_image": image_emb}
            if text_emb is not None:
                vector["dense_text"] = text_emb
            if sparse_result is not None:
                vector["sparse"] = SparseVector(
                    indices=sparse_result.indices.tolist(),
                    values=sparse_result.values.tolist(),
                )

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
                    },
                )],
            )
            logger.debug(f"Persisted frame point: {frame_path} (segment={segment_index})")

    # ── Internal: embeddings ──────────────────────────────────────────────────

    def _compute_image_embeddings(self, frame_paths: List[str]) -> List[List[float]]:
        inputs = []
        for fp in frame_paths:
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                inputs.append({"image": f"data:image/jpeg;base64,{b64}"})

        if not inputs:
            return []

        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        response = dashscope.MultiModalEmbedding.call(
            api_key=self._api_key,
            model=self._model,
            input=inputs,
            timeout=20,
        )

        if response.status_code != 200:
            raise ValueError(f"DashScope API error: {response.message}")

        raw = response.output.get("embeddings", [])
        if not raw:
            raise ValueError("Invalid embeddings response")

        return [emb["embedding"] for emb in raw]

    def _compute_text_dense_embedding(self, text: str) -> List[float]:
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        response = dashscope.MultiModalEmbedding.call(
            api_key=self._api_key,
            model=self._model,
            input=[{"text": text}],
            timeout=20,
        )

        if response.status_code != 200:
            raise ValueError(f"DashScope text embedding error: {response.message}")

        raw = response.output.get("embeddings", [])
        if not raw:
            raise ValueError("Invalid text embedding response")

        return raw[0]["embedding"]

    def _preprocess_transcript(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
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
                    "dense_image": VectorParams(size=vector_size, distance=Distance.COSINE),
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
