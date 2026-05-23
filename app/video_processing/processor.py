"""
Video Processing
================
HLS-only segment processing utilities.

This module no longer downloads or splits videos. It assumes HLS
segments already exist on disk and provides:
    - per-segment frame extraction
    - per-segment audio extraction
    - HLS playlist watching (HLSSegmentWatcher)
"""

import os
import time
import asyncio
import logging
import subprocess
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Generator, AsyncGenerator, Callable
from enum import Enum

import math

import cv2
import numpy as np
import imagehash
from PIL import Image

from app.config.config import TEMPORARY_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class VideoSegment:
    """Represents a segment of a video with associated metadata."""
    index: int
    start_time: float          # seconds
    end_time: float            # seconds
    video_path: str            # path to the segment clip
    audio_path: Optional[str] = None  # path to extracted audio (.wav)
    transcript: Optional[str] = None  # STT result for this segment
    frames: List[str] = field(default_factory=list)  # extracted frame paths
    frame_timestamps: List[float] = field(default_factory=list)  # timestamps of extracted frames

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ──────────────────────────────────────────────
# Video Streaming Processor
# ──────────────────────────────────────────────

class VideoStreamingProcessor:
    """
    Processes existing video segments by:
    1. Extracting keyframes from each segment
    2. Extracting audio from each segment
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        shortest_edge: int = 448,
        jpeg_quality: int = 90,
        frame_extraction_fps: float = 1.0,
        k_frames: int = 3,
    ):
        self.output_dir = output_dir or os.path.join(TEMPORARY_DIR, "video_streaming")
        self.shortest_edge = shortest_edge
        self.jpeg_quality = jpeg_quality
        self.frame_extraction_fps = frame_extraction_fps
        self.k_frames = k_frames

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "audio"), exist_ok=True)

    # ── Video metadata ────────────────────────

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                duration = float(info.get("format", {}).get("duration", 0))
                video_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
                audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]

                width = int(video_streams[0].get("width", 0)) if video_streams else 0
                height = int(video_streams[0].get("height", 0)) if video_streams else 0
                fps_str = video_streams[0].get("r_frame_rate", "30/1") if video_streams else "30/1"

                # Parse fps fraction
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else 30.0
                else:
                    fps = float(fps_str)

                return {
                    "duration": duration,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "has_audio": len(audio_streams) > 0,
                }
        except Exception as e:
            logger.warning(f"ffprobe failed: {e}")

        # Fallback: use OpenCV for basic info
        cap = cv2.VideoCapture(video_path)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        info = {
            "duration": duration,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "has_audio": False,  # OpenCV can't detect audio
        }
        cap.release()
        return info

    # ── Frame extraction ──────────────────────

    def extract_frames_from_segment(self, segment: VideoSegment) -> List[str]:
        """
        Extract up to k_frames content-aware keyframes from a video segment.

        Candidates are subsampled at frame_extraction_fps from the all-frames
        buffer. pHash + greedy max-min (farthest-point sampling) selects k
        maximally-distinct frames, seeded from the candidate nearest the segment
        midpoint. When fewer candidates exist than k, all are returned as-is.
        """
        cap = cv2.VideoCapture(segment.video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open segment video: {segment.video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Compute resize scale (use reported dims; fall back gracefully if zero)
        min_dim = min(width, height) if width and height else self.shortest_edge
        scale = self.shortest_edge / min_dim if min_dim > self.shortest_edge else 1.0
        new_w = int(width * scale)
        new_h = int(height * scale)
        new_w = new_w if new_w % 2 == 0 else new_w - 1
        new_h = new_h if new_h % 2 == 0 else new_h - 1

        frame_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        # CAP_PROP_FRAME_COUNT is unreliable for HLS .ts — read all frames
        all_frames: List[tuple] = []  # (frame_ndarray, original_index)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append((frame, idx))
            idx += 1

        cap.release()

        if not all_frames:
            logger.warning(f"Segment {segment.index} has no readable frames, skipping.")
            return []

        # 1fps subsampling — build candidate list from existing buffer
        step = max(1, math.ceil(video_fps / self.frame_extraction_fps))
        candidates = all_frames[::step]  # list of (frame_ndarray, original_index)
        n_candidates = len(candidates)
        k = min(self.k_frames, n_candidates)

        if n_candidates <= self.k_frames:
            # Fewer candidates than k — use all without pHash (R4 / AE1)
            selected = candidates
        else:
            # Compute pHash for each candidate
            phashed = []
            for frame, orig_idx in candidates:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ph = imagehash.phash(Image.fromarray(rgb))
                phashed.append((frame, orig_idx, ph))

            # Greedy max-min: seed from candidate nearest the segment midpoint
            n_total = len(all_frames)
            midpoint = n_total // 2
            seed_i = min(range(len(phashed)), key=lambda i: abs(phashed[i][1] - midpoint))
            selected_indices = {seed_i}
            selected_hashes = [phashed[seed_i][2]]

            for _ in range(k - 1):
                best_i, best_dist = -1, -1
                for i, (_, _, ph) in enumerate(phashed):
                    if i in selected_indices:
                        continue
                    min_dist = min(ph - sh for sh in selected_hashes)
                    if min_dist > best_dist:
                        best_dist = min_dist
                        best_i = i
                if best_i == -1:
                    break
                selected_indices.add(best_i)
                selected_hashes.append(phashed[best_i][2])

            # Sort temporally by original frame index
            selected = sorted(
                [(phashed[i][0], phashed[i][1]) for i in selected_indices],
                key=lambda x: x[1],
            )

        # Save selected frames as JPEG
        frame_paths: List[str] = []
        frame_timestamps: List[float] = []
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        for j, (frame, orig_idx) in enumerate(selected):
            if scale != 1.0:
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            frame_filename = f"seg{segment.index:03d}_{timestamp_str}_{j:02d}.jpg"
            frame_path = os.path.join(frame_dir, frame_filename)
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

            frame_paths.append(frame_path)
            frame_timestamps.append(segment.start_time + orig_idx / video_fps)

        segment.frames = frame_paths
        segment.frame_timestamps = frame_timestamps
        logger.info(f"Extracted {len(frame_paths)} frames from segment {segment.index}")
        return frame_paths

    # ── Audio extraction ──────────────────────

    def extract_audio_from_segment(self, segment: VideoSegment) -> Optional[str]:
        """
        Extract audio track from a video segment as WAV using ffmpeg.

        :param segment: VideoSegment to extract audio from
        :return: Path to the extracted .wav file, or None if no audio
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(
            self.output_dir, "audio",
            f"audio_seg{segment.index}_{timestamp}.wav"
        )

        cmd = [
            "ffmpeg", "-y", "-i", segment.video_path,
            "-vn",                    # no video
            "-acodec", "pcm_s16le",   # 16-bit PCM
            "-ar", "16000",           # 16kHz sample rate (good for STT)
            "-ac", "1",               # mono
            audio_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and os.path.exists(audio_path):
                segment.audio_path = audio_path
                logger.info(f"Extracted audio from segment {segment.index}: {audio_path}")
                return audio_path
            logger.warning(f"ffmpeg audio extraction failed for segment {segment.index}: {result.stderr[:300]}")
        except Exception as e:
            logger.warning(f"Audio extraction error for segment {segment.index}: {e}")

        return None

class StreamEvent(Enum):
    """Events emitted by the HLS segment watcher."""
    STREAM_START = "stream_start"
    SEGMENT_READY = "segment_ready"
    SEGMENT_PROCESSED = "segment_processed"
    STREAM_END = "stream_end"


@dataclass
class StreamUpdate:
    """A single update emitted during HLS streaming."""
    event: StreamEvent
    segment: Optional[VideoSegment] = None
    elapsed_real_time: float = 0.0        # wall-clock seconds since stream started
    elapsed_video_time: float = 0.0       # video-time seconds since start
    segments_completed: int = 0
    total_segments: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# HLS Segment Watcher
# ──────────────────────────────────────────────

class HLSSegmentWatcher:
    """
    Watches an HLS playlist and processes segments as they appear.

    This does not generate HLS; it only consumes a local playlist and
    segment files produced elsewhere (e.g., ffmpeg -f hls).
    """

    def __init__(
        self,
        processor: VideoStreamingProcessor,
        playlist_path: str,
        poll_interval: float = 0.5,
        start_from_current: bool = True,
        extract_frames: bool = True,
        extract_audio: bool = False,
        segment_root: Optional[str] = None,
        on_segment: Optional[Callable[[VideoSegment], None]] = None,
    ):
        self.processor = processor
        self.playlist_path = playlist_path
        self.poll_interval = max(poll_interval, 0.1)
        self.start_from_current = start_from_current
        self.extract_frames = extract_frames
        self.extract_audio = extract_audio
        self.segment_root = segment_root
        self.on_segment = on_segment

    def stream(self) -> Generator[StreamUpdate, None, None]:
        seen_paths = set()
        start_wall = time.monotonic()
        processed = 0

        yield StreamUpdate(
            event=StreamEvent.STREAM_START,
            total_segments=0,
            metadata={"playlist": self.playlist_path},
        )

        # Wait for ffmpeg to create the playlist when starting fresh
        _PLAYLIST_WAIT_TIMEOUT = 60
        _waited = 0.0
        while not os.path.exists(self.playlist_path):
            if _waited >= _PLAYLIST_WAIT_TIMEOUT:
                raise TimeoutError(f"Playlist not found after {_PLAYLIST_WAIT_TIMEOUT}s: {self.playlist_path}")
            time.sleep(self.poll_interval)
            _waited += self.poll_interval

        if self.start_from_current:
            media_seq, segments, _ = _parse_hls_playlist(self.playlist_path)
            for i, (uri, _dur) in enumerate(segments):
                seg_path = _resolve_hls_segment_path(
                    self.playlist_path, uri, self.segment_root
                )
                seen_paths.add(seg_path)

        while True:
            media_seq, segments, endlist = _parse_hls_playlist(self.playlist_path)
            cursor = 0.0

            for i, (uri, dur) in enumerate(segments):
                seg_path = _resolve_hls_segment_path(
                    self.playlist_path, uri, self.segment_root
                )
                start_time = cursor
                end_time = cursor + max(dur, 0.0)
                cursor = end_time

                if seg_path in seen_paths:
                    continue
                if not os.path.exists(seg_path):
                    continue

                seen_paths.add(seg_path)
                segment = VideoSegment(
                    index=media_seq + i,
                    start_time=start_time,
                    end_time=end_time,
                    video_path=seg_path,
                )

                yield StreamUpdate(
                    event=StreamEvent.SEGMENT_READY,
                    segment=segment,
                    elapsed_real_time=time.monotonic() - start_wall,
                    elapsed_video_time=end_time,
                    segments_completed=processed,
                )

                if self.extract_frames:
                    self.processor.extract_frames_from_segment(segment)
                if self.extract_audio:
                    self.processor.extract_audio_from_segment(segment)

                if self.on_segment:
                    try:
                        self.on_segment(segment)
                    except Exception as cb_err:
                        logger.warning(f"on_segment callback error: {cb_err}")

                processed += 1
                yield StreamUpdate(
                    event=StreamEvent.SEGMENT_PROCESSED,
                    segment=segment,
                    elapsed_real_time=time.monotonic() - start_wall,
                    elapsed_video_time=end_time,
                    segments_completed=processed,
                    metadata={
                        "frames_extracted": len(segment.frames),
                        "has_audio": segment.audio_path is not None,
                    },
                )

            if endlist:
                yield StreamUpdate(
                    event=StreamEvent.STREAM_END,
                    elapsed_real_time=time.monotonic() - start_wall,
                    elapsed_video_time=cursor,
                    segments_completed=processed,
                )
                break

            time.sleep(self.poll_interval)

    async def astream(self) -> AsyncGenerator[StreamUpdate, None]:
        seen_paths = set()
        start_wall = time.monotonic()
        processed = 0

        yield StreamUpdate(
            event=StreamEvent.STREAM_START,
            total_segments=0,
            metadata={"playlist": self.playlist_path},
        )

        # Wait for ffmpeg to create the playlist when starting fresh
        _PLAYLIST_WAIT_TIMEOUT = 60
        _waited = 0.0
        while not os.path.exists(self.playlist_path):
            if _waited >= _PLAYLIST_WAIT_TIMEOUT:
                raise TimeoutError(f"Playlist not found after {_PLAYLIST_WAIT_TIMEOUT}s: {self.playlist_path}")
            await asyncio.sleep(self.poll_interval)
            _waited += self.poll_interval

        if self.start_from_current:
            media_seq, segments, _ = _parse_hls_playlist(self.playlist_path)
            for i, (uri, _dur) in enumerate(segments):
                seg_path = _resolve_hls_segment_path(
                    self.playlist_path, uri, self.segment_root
                )
                seen_paths.add(seg_path)

        while True:
            media_seq, segments, endlist = _parse_hls_playlist(self.playlist_path)
            cursor = 0.0

            for i, (uri, dur) in enumerate(segments):
                seg_path = _resolve_hls_segment_path(
                    self.playlist_path, uri, self.segment_root
                )
                start_time = cursor
                end_time = cursor + max(dur, 0.0)
                cursor = end_time

                if seg_path in seen_paths:
                    continue
                if not os.path.exists(seg_path):
                    continue

                seen_paths.add(seg_path)
                segment = VideoSegment(
                    index=media_seq + i,
                    start_time=start_time,
                    end_time=end_time,
                    video_path=seg_path,
                )

                yield StreamUpdate(
                    event=StreamEvent.SEGMENT_READY,
                    segment=segment,
                    elapsed_real_time=time.monotonic() - start_wall,
                    elapsed_video_time=end_time,
                    segments_completed=processed,
                )

                loop = asyncio.get_event_loop()
                if self.extract_frames:
                    await loop.run_in_executor(
                        None, self.processor.extract_frames_from_segment, segment
                    )
                if self.extract_audio:
                    await loop.run_in_executor(
                        None, self.processor.extract_audio_from_segment, segment
                    )

                if self.on_segment:
                    try:
                        self.on_segment(segment)
                    except Exception as cb_err:
                        logger.warning(f"on_segment callback error: {cb_err}")

                processed += 1
                yield StreamUpdate(
                    event=StreamEvent.SEGMENT_PROCESSED,
                    segment=segment,
                    elapsed_real_time=time.monotonic() - start_wall,
                    elapsed_video_time=end_time,
                    segments_completed=processed,
                    metadata={
                        "frames_extracted": len(segment.frames),
                        "has_audio": segment.audio_path is not None,
                    },
                )

            if endlist:
                yield StreamUpdate(
                    event=StreamEvent.STREAM_END,
                    elapsed_real_time=time.monotonic() - start_wall,
                    elapsed_video_time=cursor,
                    segments_completed=processed,
                )
                break

            await asyncio.sleep(self.poll_interval)


# ── Utilities ─────────────────────────────────

def _parse_hls_playlist(playlist_path: str) -> Tuple[int, List[Tuple[str, float]], bool]:
    if not os.path.exists(playlist_path):
        return 0, [], False

    media_seq = 0
    segments: List[Tuple[str, float]] = []
    endlist = False
    current_duration = None

    try:
        with open(playlist_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
    except Exception:
        return 0, [], False

    for line in lines:
        if not line:
            continue
        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            try:
                media_seq = int(line.split(":", 1)[1])
            except ValueError:
                media_seq = 0
            continue
        if line.startswith("#EXTINF:"):
            try:
                current_duration = float(line.split(":", 1)[1].split(",", 1)[0])
            except ValueError:
                current_duration = 0.0
            continue
        if line.startswith("#EXT-X-ENDLIST"):
            endlist = True
            continue
        if line.startswith("#"):
            continue

        segments.append((line, float(current_duration or 0.0)))
        current_duration = None

    return media_seq, segments, endlist


def _resolve_hls_segment_path(
    playlist_path: str,
    segment_uri: str,
    segment_root: Optional[str] = None,
) -> str:
    if os.path.isabs(segment_uri):
        return segment_uri
    if segment_root:
        return os.path.join(segment_root, segment_uri)
    return os.path.join(os.path.dirname(playlist_path), segment_uri)
