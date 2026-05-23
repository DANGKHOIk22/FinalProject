"""
Match Indexing Service
======================
Background processing for HLS sessions:
- Extract frames per segment
- Embed frames
- Index into Qdrant
"""

import logging
from typing import Callable, Optional

from app.config.settings import settings
from app.video_processing import VideoStreamingProcessor, HLSSegmentWatcher, StreamEvent
from app.video_processing.segment_index import IncrementalSegmentIndexer
from app.video_processing.speech_to_text import SpeechToTextService

logger = logging.getLogger(__name__)


class MatchIndexingService:
    """Service that processes an HLS session and indexes frame embeddings."""

    def __init__(self, dashscope_api_key: Optional[str] = None):
        self._dashscope_api_key = dashscope_api_key or settings.DASHSCOPE_API_KEY
        self._stt = SpeechToTextService()

    def process_hls_session(
        self,
        video_id: str,
        hls_dir: str,
        playlist_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> int:
        from qdrant_client import QdrantClient

        qdrant_client = QdrantClient(
            url=settings.QDRANT_URL ,
            api_key=settings.QDRANT_API_KEY,
            check_compatibility=False
        )
        collection = settings.QDRANT_HLS_COLLECTION_NAME or "hls_frame_index"

        processor = VideoStreamingProcessor(output_dir=output_dir)
        watcher = HLSSegmentWatcher(
            processor=processor,
            playlist_path=playlist_path,
            extract_frames=True,
            extract_audio=True,
            start_from_current=False,
            segment_root=output_dir,
        )
        indexer = IncrementalSegmentIndexer(
            dashscope_api_key=self._dashscope_api_key,
            qdrant_client=qdrant_client,
            qdrant_collection=collection,
            video_id=video_id,
        )

        processed = 0
        for update in watcher.stream():
            if update.event == StreamEvent.SEGMENT_PROCESSED and update.segment:
                seg = update.segment
                if seg.audio_path:
                    try:
                        seg.transcript = self._stt.transcribe(seg.audio_path)
                    except Exception as e:
                        logger.warning(f"STT failed for segment {seg.index}: {e}")
                        seg.transcript = ""
                indexer.index_segment_frames(
                    frame_paths=seg.frames,
                    transcript=seg.transcript or "",
                    segment_index=seg.index,
                    timestamp=seg.start_time,
                )
                processed += 1
                if progress_callback:
                    progress_callback(processed)
            elif update.event == StreamEvent.STREAM_END:
                break

        return processed
