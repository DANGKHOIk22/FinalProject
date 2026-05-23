"""
Unit tests for the HLS video processing pipeline.

Tests cover:
- VideoStreamingProcessor: video info, frame extraction, audio extraction
- SpeechToTextService: backend factory, stub backend, segment transcription
- IncrementalSegmentIndexer: no-key fallback, accumulation, captioning happy/failure paths
- FrameSelectionTool: HLS-only validation, TextEmbedding usage
"""

import json
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_video(temp_dir):
    """Create a minimal test video file using OpenCV."""
    import cv2
    import numpy as np

    video_path = os.path.join(temp_dir, "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (320, 240))

    # Write 90 frames (3 seconds at 30fps)
    for i in range(90):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def sample_audio(temp_dir):
    """Create a minimal WAV file for STT testing."""
    import wave

    audio_path = os.path.join(temp_dir, "test_audio.wav")
    with wave.open(audio_path, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        samples = b"\x00\x00" * 16000  # 1 second of silence
        wav.writeframes(samples)
    return audio_path


@pytest.fixture
def fake_frames(temp_dir):
    """Create minimal JPEG files for use as mock frame inputs."""
    import cv2
    import numpy as np

    paths = []
    for i in range(3):
        path = os.path.join(temp_dir, f"frame_{i:02d}.jpg")
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


# ──────────────────────────────────────────────
# VideoStreamingProcessor tests
# ──────────────────────────────────────────────

class TestVideoStreamingProcessor:

    def test_init_creates_directories(self, temp_dir):
        from app.video_processing.processor import VideoStreamingProcessor

        processor = VideoStreamingProcessor(output_dir=temp_dir)
        assert os.path.isdir(os.path.join(temp_dir, "frames"))
        assert os.path.isdir(os.path.join(temp_dir, "audio"))

    def test_get_video_info_with_opencv(self, sample_video, temp_dir):
        from app.video_processing.processor import VideoStreamingProcessor

        processor = VideoStreamingProcessor(output_dir=temp_dir)
        info = processor.get_video_info(sample_video)

        assert info["width"] == 320
        assert info["height"] == 240
        assert info["fps"] > 0
        assert info["duration"] > 0

    def test_extract_frames_from_segment(self, sample_video, temp_dir):
        from app.video_processing.processor import (
            VideoStreamingProcessor,
            VideoSegment,
        )

        processor = VideoStreamingProcessor(
            output_dir=temp_dir,
            frame_extraction_fps=10.0,
            k_frames=5,
        )

        segment = VideoSegment(
            index=0,
            start_time=0.0,
            end_time=3.0,
            video_path=sample_video,
        )

        frame_paths = processor.extract_frames_from_segment(segment)
        assert len(frame_paths) > 0
        assert all(os.path.exists(p) for p in frame_paths)
        assert all(p.endswith(".jpg") for p in frame_paths)
        assert segment.frames == frame_paths
        assert len(segment.frame_timestamps) == len(frame_paths)

    def test_extract_audio_from_segment_no_audio(self, sample_video, temp_dir):
        """OpenCV-generated video has no audio track; ffmpeg should handle gracefully."""
        from app.video_processing.processor import (
            VideoStreamingProcessor,
            VideoSegment,
        )

        processor = VideoStreamingProcessor(output_dir=temp_dir)
        segment = VideoSegment(
            index=0,
            start_time=0.0,
            end_time=3.0,
            video_path=sample_video,
        )

        result = processor.extract_audio_from_segment(segment)
        if result is not None:
            assert os.path.exists(result)


# ──────────────────────────────────────────────
# SpeechToTextService tests
# ──────────────────────────────────────────────

class TestSpeechToTextService:

    def test_transcribe_returns_string_on_api_failure(self, sample_audio):
        """When DashScope call fails, transcribe returns empty string (no raise)."""
        from unittest.mock import patch
        from app.video_processing.speech_to_text import SpeechToTextService

        with patch("app.video_processing.speech_to_text.dashscope.MultiModalConversation") as MockASR:
            MockASR.call.side_effect = Exception("API error")
            stt = SpeechToTextService(api_key="test-key")
            result = stt.transcribe(sample_audio)
        assert isinstance(result, str)
        assert result == ""

    def test_transcribe_batch_returns_list(self, sample_audio):
        """transcribe_batch returns one result per input path."""
        from unittest.mock import patch
        from app.video_processing.speech_to_text import SpeechToTextService

        with patch("app.video_processing.speech_to_text.dashscope.MultiModalConversation") as MockASR:
            MockASR.call.side_effect = Exception("API error")
            stt = SpeechToTextService(api_key="test-key")
            results = stt.transcribe_batch([sample_audio, sample_audio])
        assert len(results) == 2

    def test_transcribe_segments_sets_transcript(self, sample_audio):
        """transcribe_segments populates transcript on segments with audio; empty string for those without."""
        from unittest.mock import patch, MagicMock
        from app.video_processing.speech_to_text import SpeechToTextService
        from app.video_processing.processor import VideoSegment

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output.choices[0].message.content = "commentary text"

        with patch("app.video_processing.speech_to_text.dashscope.MultiModalConversation") as MockASR:
            MockASR.call.return_value = mock_response
            stt = SpeechToTextService(api_key="test-key")
            segments = [
                VideoSegment(index=0, start_time=0, end_time=30, video_path="/tmp/seg.mp4", audio_path=sample_audio),
                VideoSegment(index=1, start_time=30, end_time=60, video_path="/tmp/seg.mp4", audio_path=None),
            ]
            results = stt.transcribe_segments(segments)

        assert len(results) == 2
        assert segments[0].transcript == "commentary text"
        assert segments[1].transcript == ""

    def test_no_api_key_raises(self, monkeypatch):
        """Missing DASHSCOPE_API_KEY raises ValueError at construction."""
        from app.video_processing.speech_to_text import SpeechToTextService
        from app.config import settings

        monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "")
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            SpeechToTextService(api_key=None)

    def test_file_not_found_raises(self):
        """transcribe raises FileNotFoundError for a missing audio path."""
        from app.video_processing.speech_to_text import SpeechToTextService

        stt = SpeechToTextService(api_key="test-key")
        with pytest.raises(FileNotFoundError):
            stt.transcribe("/nonexistent/path/audio.wav")


# ──────────────────────────────────────────────
# IncrementalSegmentIndexer tests
# ──────────────────────────────────────────────

def _make_mock_caption_response(captions: list[dict]) -> MagicMock:
    """Build a mock OpenAI chat completion response returning a JSON caption array."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(captions)
    return mock_response


def _make_mock_embedding_response(vector: list[float]) -> MagicMock:
    """Build a mock dashscope TextEmbedding response."""
    from http import HTTPStatus
    mock_resp = MagicMock()
    mock_resp.status_code = HTTPStatus.OK
    mock_resp.output = {"embeddings": [{"embedding": vector, "text_index": 0}]}
    return mock_resp


class TestIncrementalSegmentIndexer:

    def test_no_api_key_tracks_frames_without_indexing(self, sample_video, temp_dir):
        """Without API key, frames are tracked but no API calls or Qdrant upserts occur."""
        from app.video_processing.segment_index import IncrementalSegmentIndexer
        from app.video_processing.processor import VideoStreamingProcessor, VideoSegment

        processor = VideoStreamingProcessor(output_dir=temp_dir, frame_extraction_fps=10.0, k_frames=5)
        seg = VideoSegment(index=0, start_time=0, end_time=3, video_path=sample_video)
        processor.extract_frames_from_segment(seg)

        mock_qdrant = MagicMock()
        indexer = IncrementalSegmentIndexer(
            dashscope_api_key="",
            qdrant_client=mock_qdrant,
            video_id="test-video",
        )
        indexer.index_segment_frames(seg.frames, transcript="goal", segment_index=0, timestamp=0.0)

        assert indexer.total_indexed == len(seg.frames)
        assert indexer.frame_paths == seg.frames
        mock_qdrant.upsert.assert_not_called()

    def test_empty_indexer_has_zero_total(self):
        """A freshly created indexer has zero total_indexed."""
        from app.video_processing.segment_index import IncrementalSegmentIndexer

        indexer = IncrementalSegmentIndexer(dashscope_api_key="")
        assert indexer.total_indexed == 0
        assert indexer.frame_paths == []

    def test_incremental_accumulation(self, sample_video, temp_dir):
        """Calling index_segment_frames multiple times accumulates frame_paths."""
        from app.video_processing.segment_index import IncrementalSegmentIndexer
        from app.video_processing.processor import VideoStreamingProcessor, VideoSegment

        processor = VideoStreamingProcessor(output_dir=temp_dir, frame_extraction_fps=10.0, k_frames=3)
        seg0 = VideoSegment(index=0, start_time=0, end_time=1.5, video_path=sample_video)
        seg1 = VideoSegment(index=1, start_time=1.5, end_time=3.0, video_path=sample_video)
        processor.extract_frames_from_segment(seg0)
        processor.extract_frames_from_segment(seg1)

        indexer = IncrementalSegmentIndexer(dashscope_api_key="")
        indexer.index_segment_frames(seg0.frames, transcript="pass", segment_index=0, timestamp=0.0)
        after_first = indexer.total_indexed

        indexer.index_segment_frames(seg1.frames, transcript="shot", segment_index=1, timestamp=1.5)
        after_second = indexer.total_indexed

        assert after_second > after_first
        assert after_second == len(seg0.frames) + len(seg1.frames)

    def test_empty_frame_list_no_op(self):
        """index_segment_frames with empty list does nothing."""
        from app.video_processing.segment_index import IncrementalSegmentIndexer

        mock_qdrant = MagicMock()
        indexer = IncrementalSegmentIndexer(
            dashscope_api_key="key",
            qdrant_client=mock_qdrant,
            video_id="vid",
        )
        indexer.index_segment_frames([], transcript="", segment_index=0, timestamp=0.0)

        assert indexer.total_indexed == 0
        mock_qdrant.upsert.assert_not_called()

    def test_caption_batch_happy_path(self, fake_frames):
        """With mocked captioning + embedding, upsert is called once per frame with correct schema."""
        from app.video_processing.segment_index import IncrementalSegmentIndexer

        fake_vector = [0.1] * 1024
        captions_payload = [
            {"image_index": 0, "description": "A player in red is running."},
            {"image_index": 1, "description": "A referee in black is gesturing."},
            {"image_index": 2, "description": "A goalkeeper in yellow is jumping."},
        ]

        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value.collections = []

        with patch("app.video_processing.segment_index.OpenAI") as MockOpenAI, \
             patch("app.video_processing.segment_index.dashscope.TextEmbedding") as MockTextEmb:

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _make_mock_caption_response(captions_payload)
            MockOpenAI.return_value = mock_client
            MockTextEmb.call.return_value = _make_mock_embedding_response(fake_vector)

            indexer = IncrementalSegmentIndexer(
                dashscope_api_key="test-key",
                qdrant_client=mock_qdrant,
                video_id="vid-001",
            )
            indexer.index_segment_frames(
                fake_frames,
                transcript="great goal",
                segment_index=0,
                timestamp=0.0,
            )

        assert mock_qdrant.upsert.call_count == len(fake_frames)

        for call in mock_qdrant.upsert.call_args_list:
            point = call.kwargs["points"][0]
            assert "dense_caption" in point.vector
            assert "dense_text" in point.vector
            assert "caption" in point.payload
            assert "frame_path" in point.payload

    def test_caption_api_failure_skips_frames(self, fake_frames):
        """When captioning API raises, no frames are indexed but frame_paths still tracks them."""
        from app.video_processing.segment_index import IncrementalSegmentIndexer

        mock_qdrant = MagicMock()

        with patch("app.video_processing.segment_index.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API unavailable")
            MockOpenAI.return_value = mock_client

            indexer = IncrementalSegmentIndexer(
                dashscope_api_key="test-key",
                qdrant_client=mock_qdrant,
                video_id="vid-001",
            )
            indexer.index_segment_frames(
                fake_frames,
                transcript="corner kick",
                segment_index=0,
                timestamp=10.0,
            )

        mock_qdrant.upsert.assert_not_called()
        assert indexer.total_indexed == len(fake_frames)
        assert indexer.frame_paths == fake_frames

    def test_partial_caption_skips_missing_indices(self, fake_frames):
        """When only some frame captions are returned, only those frames are indexed."""
        from app.video_processing.segment_index import IncrementalSegmentIndexer

        fake_vector = [0.1] * 1024
        # Only caption for index 0 — indices 1 and 2 are missing
        partial_captions = [{"image_index": 0, "description": "A player celebrating."}]

        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value.collections = []

        with patch("app.video_processing.segment_index.OpenAI") as MockOpenAI, \
             patch("app.video_processing.segment_index.dashscope.TextEmbedding") as MockTextEmb:

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _make_mock_caption_response(partial_captions)
            MockOpenAI.return_value = mock_client
            MockTextEmb.call.return_value = _make_mock_embedding_response(fake_vector)

            indexer = IncrementalSegmentIndexer(
                dashscope_api_key="test-key",
                qdrant_client=mock_qdrant,
                video_id="vid-001",
            )
            indexer.index_segment_frames(
                fake_frames,
                transcript="penalty",
                segment_index=1,
                timestamp=30.0,
            )

        assert mock_qdrant.upsert.call_count == 1
        point = mock_qdrant.upsert.call_args.kwargs["points"][0]
        assert point.payload["caption"] == "A player celebrating."


# ──────────────────────────────────────────────
# FrameSelection tests
# ──────────────────────────────────────────────

class TestFrameSelectionTool:

    def test_requires_video_id(self, monkeypatch):
        """Calling _run without video_id raises ValueError."""
        from app.config import settings
        from app.soccer_agent.toolbox.frame_selection import FrameSelectionTool

        monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test")
        tool = FrameSelectionTool()
        result_msg, result_artifact = tool._run(query="goal", video_id=None)
        assert "video_id is required" in result_msg
        assert result_artifact is None

    def test_embed_text_uses_text_embedding_v4(self, monkeypatch):
        """_embed_text calls dashscope.TextEmbedding with text-embedding-v4."""
        from http import HTTPStatus
        from app.config import settings
        from app.soccer_agent.toolbox.frame_selection import FrameSelectionTool

        monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test")
        tool = FrameSelectionTool()

        fake_vector = [0.5] * 1024
        mock_resp = MagicMock()
        mock_resp.status_code = HTTPStatus.OK
        mock_resp.output = {"embeddings": [{"embedding": fake_vector, "text_index": 0}]}

        with patch("app.soccer_agent.toolbox.frame_selection.dashscope.TextEmbedding") as MockTextEmb:
            MockTextEmb.call.return_value = mock_resp
            result = tool._embed_text("great goal")

        MockTextEmb.call.assert_called_once()
        call_kwargs = MockTextEmb.call.call_args.kwargs
        assert call_kwargs.get("model") == "text-embedding-v4"
        assert result == fake_vector
