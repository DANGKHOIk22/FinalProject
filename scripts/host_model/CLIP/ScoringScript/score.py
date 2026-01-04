import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Mitigate CUDA fragmentation on long-running containers
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")

logger = logging.getLogger(__name__)


def _parse_payload(data: Any) -> Dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8", errors="replace")
    if isinstance(data, str):
        data = data.strip()
        if not data:
            return {}
        return json.loads(data)
    try:
        return dict(data)
    except Exception:
        return {}


def _extract_frames(payload: Dict[str, Any]) -> List[str]:
    """
    Extract frames from the payload.
    :param payload: The input payload.
    :return: A list of base64-encoded frame strings.
    """
    frames = payload.get("images") or payload.get("frames")
    if not frames or not isinstance(frames, list):
        raise ValueError('Field "images" (or "frames") must be a non-empty list of base64 strings')
    for idx, frame in enumerate(frames):
        if not isinstance(frame, str):
            raise TypeError(f"Frame at index {idx} is not a base64 string")
    return frames


def _extract_query(payload: Dict[str, Any]) -> str:
    """
    Extract the query string from the payload.
    :param payload: The input payload.
    :return: The query string.
    """
    query = payload.get("query") or payload.get("text")
    if not query or not isinstance(query, str):
        raise ValueError('Field "query" must be a non-empty string')
    return query


def _base64_to_pil(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _pil_to_base64(img: Image.Image) -> str:
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def init():
    """Load CLIP model and processor once per container."""
    model_dir = os.getenv("AZUREML_MODEL_DIR", "")
    if not model_dir:
        raise EnvironmentError("AZUREML_MODEL_DIR is not set")

    base_model_dir = Path(model_dir)
    model_dir_path = base_model_dir / ".clip" if (base_model_dir / ".clip").exists() else base_model_dir

    global device, processor, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(model_dir_path)
    model = CLIPModel.from_pretrained(model_dir_path).to(device)
    model.eval()
    logger.info("CLIP initialized on device %s", device)


def run(raw_data: Any):
    """Select frames most relevant to the query using CLIP similarity.
     :param raw_data: The input payload containing frames and query.
     :return: A dictionary with selected frame indices and their scores."""
    try:
        # Validate and parse input payload
        payload = _parse_payload(raw_data)
        frame_b64_list = _extract_frames(payload)
        query = _extract_query(payload)

        # Clamp topk and threshold to sensible ranges
        topk = int(payload.get("topk", 5))
        topk = max(1, min(topk, 10))
        threshold = float(payload.get("threshold", 0.0))
        threshold = 0.0 if threshold < 0.0 or threshold > 1.0 else threshold

        # Convert base64 strings to PIL Images
        images = [_base64_to_pil(frame_b64) for frame_b64 in frame_b64_list]

        # Prepare inputs for CLIP model
        inputs = processor(
            text=[query],
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image #  This represents the image-text similarity scores. The shape is (num_images, num_texts)

        # Compute similarities
        similarities = torch.nn.functional.softmax(logits_per_image, dim=0).squeeze(1)

        # Adjust topk if there are fewer images than topk. And get topk results
        topk = min(topk, similarities.shape[-1])
        top_scores, top_indices = torch.topk(similarities, k=topk)

        selected_frames: List[int] = []
        selected_scores: List[float] = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            if score < threshold:
                continue
            selected_frames.append(int(idx))
            selected_scores.append(float(score))

        return {
            "success": True,
            "selected_frames": selected_frames,
            "scores": selected_scores,
            "meta": {
                "topk": topk,
                "threshold": threshold,
                "query": query,
                "device": str(device),
            },
        }
    except Exception as exc:  
        logger.exception("CLIP scoring failed")
        return {"success": False, "error": str(exc)}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
