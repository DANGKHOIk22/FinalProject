import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from accelerate import Accelerator
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
)

# Mitigate CUDA fragmentation on long-running containers
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _parse_payload(data: Any) -> Dict[str, Any]:
    """
    Parse the input payload into a dictionary.
    :param data: The input payload.
    :return: A dictionary representation of the payload.
    """

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


def _extract_task(payload: Dict[str, Any]) -> str:
    """
    Extract the task from the payload.
    :param payload: The input payload.
    :return: The task string.
    """
    task = payload.get("task") or payload.get("mode")
    if not task or not isinstance(task, str):
        raise ValueError('Field "task" must be provided ("clip" or "groundingdino")')
    task = task.strip().lower()
    if task not in {"clip", "groundingdino"}:
        raise ValueError('Field "task" must be either "clip" or "groundingdino"')
    return task


def _extract_frames(payload: Dict[str, Any]) -> List[str]:
    """
    Extract frames from the payload if using CLIP.
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
    Extract the query string from the payload. This is used for CLIP.
    :param payload: The input payload.
    :return: The query string.
    """
    query = payload.get("query") or payload.get("text")
    if not query or not isinstance(query, str):
        raise ValueError('Field "query" must be a non-empty string')
    return query


def _extract_base64_image(payload: Dict[str, Any]) -> str:
    """
    Extract a base64-encoded image string from the payload. This is used for GroundingDINO.
    :param payload: The input payload.
    :return: The base64-encoded image string.
    """

    image_value = payload.get("image")
    if image_value is None:
        raise KeyError('Missing required field "image" in request payload')
    if isinstance(image_value, (bytes, bytearray)):
        image_value = image_value.decode("utf-8", errors="replace")
    if not isinstance(image_value, str):
        raise TypeError('Field "image" must be a base64-encoded string')

    image_value = image_value.strip()
    if image_value.startswith("data:") and "," in image_value:
        image_value = image_value.split(",", 1)[1].strip()
    if not image_value:
        raise ValueError('Field "image" is empty')
    return image_value


def _base64_to_pil(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _pil_to_base64(img: Image.Image) -> str:
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def _json_safe(obj: Any) -> Any:
    def _default(o: Any) -> Any:
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    return json.loads(json.dumps(obj, default=_default))


# Globals populated in init()
clip_device: torch.device
clip_processor: CLIPProcessor
clip_model: CLIPModel

gd_device: torch.device
gd_processor: AutoProcessor
gd_model: AutoModelForZeroShotObjectDetection


def _load_clip(model_root: Path):
    global clip_device, clip_processor, clip_model
    clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_processor = CLIPProcessor.from_pretrained(model_root)
    clip_model = CLIPModel.from_pretrained(model_root).to(clip_device)
    clip_model.eval()
    logger.info("CLIP initialized on device %s", clip_device)


def _load_groundingdino(model_root: Path):
    global gd_device, gd_processor, gd_model
    gd_device = Accelerator().device
    gd_processor = AutoProcessor.from_pretrained(model_root)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_root).to(gd_device)
    gd_model.eval()
    logger.info("GroundingDINO initialized on device %s", gd_device)


def init():
    model_dir = os.getenv("AZUREML_MODEL_DIR", "")
    if not model_dir:
        raise EnvironmentError("AZUREML_MODEL_DIR is not set")

    base_model_dir = Path(model_dir) / ".groundingdino_and_clip"
    clip_dir = base_model_dir / ".clip" if (base_model_dir / ".clip").exists() else base_model_dir
    gd_dir = base_model_dir / ".groundingdino" if (base_model_dir / ".groundingdino").exists() else base_model_dir

    _load_clip(clip_dir)
    _load_groundingdino(gd_dir)


def _run_clip(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform frame selection using CLIP
     :param payload: The input payload containing frames and query.
     :return: A dictionary with selected frame indices and scores.
    """

    # Extract frames and query
    frame_b64_list = _extract_frames(payload)
    query = _extract_query(payload)

    # Extract parameters
    topk = int(payload.get("topk", 5))
    topk = max(1, min(topk, 10))
    threshold = float(payload.get("threshold", 0.0))
    threshold = 0.0 if threshold < 0.0 or threshold > 1.0 else threshold

    # Process frames and query through CLIP
    images = [_base64_to_pil(frame_b64) for frame_b64 in frame_b64_list]
    inputs = clip_processor(
        text=[query],
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(clip_device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image

    # Compute similarities and select top-k frames
    similarities = torch.nn.functional.softmax(logits_per_image, dim=0).squeeze(1)
    topk = min(topk, similarities.shape[-1])
    top_scores, top_indices = torch.topk(similarities, k=topk)

    # Return frame indices exceeding the threshold
    selected_frames: List[int] = []
    selected_scores: List[float] = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        if score < threshold:
            continue
        selected_frames.append(int(idx))
        selected_scores.append(float(score))

    return {
        "success": True,
        "task": "clip",
        "selected_frames": selected_frames,
        "scores": selected_scores,
        "meta": {
            "topk": topk,
            "threshold": threshold,
            "query": query,
            "device": str(clip_device),
        },
    }


def _run_groundingdino(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform object detection using GroundingDINO
    :param payload: The input payload containing image and queries.
    :return: A dictionary with detected objects.
    """

    # Extract image and queries from payload
    image_b64 = _extract_base64_image(payload)
    image = _base64_to_pil(image_b64)

    text_queries = payload.get("queries") or payload.get("text")
    if isinstance(text_queries, str):
        text_queries = [text_queries]
    if not text_queries or not isinstance(text_queries, list):
        raise ValueError('Field "queries" must be a non-empty list of strings')

    # Extract parameters
    threshold = float(payload.get("threshold", 0.6))
    text_threshold = float(payload.get("text_threshold", 0.6))
    return_crops = bool(payload.get("return_crops", False))
    max_detections = payload.get("max_detections", 10)
    if max_detections is not None:
        max_detections = int(max_detections)

    # Process image and queries through GroundingDINO
    inputs = gd_processor(images=[image], text=text_queries, return_tensors="pt").to(gd_device)

    with torch.no_grad():
        outputs = gd_model(**inputs)

    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    detections = []
    for result in results:
        boxes = result["boxes"].tolist()
        scores = result["scores"].tolist()
        labels = result["labels"]
        for box, score, label in zip(boxes, scores, labels):
            if max_detections and len(detections) >= max_detections:
                break
            x_min, y_min, x_max, y_max = box
            det = {
                "label": str(label),
                "score": float(score),
                "box": {
                    "x_min": float(x_min),
                    "y_min": float(y_min),
                    "x_max": float(x_max),
                    "y_max": float(y_max),
                },
            }
            if return_crops:
                crop = image.crop((x_min, y_min, x_max, y_max))
                det["crop_base64"] = _pil_to_base64(crop)
            detections.append(det)

    return {
        "success": True,
        "task": "groundingdino",
        "detections": _json_safe(detections),
        "meta": {
            "threshold": threshold,
            "text_threshold": text_threshold,
            "num_queries": len(text_queries),
            "device": str(gd_device),
        },
    }


def run(raw_data: Any):
    """
    This function is call when the endpoint receive the request
    
    """
    try:
        payload = _parse_payload(raw_data)
        task = _extract_task(payload)

        if task == "clip":
            result = _run_clip(payload)
        else:
            result = _run_groundingdino(payload)

        return result
    except Exception as exc:
        logger.exception("CLIP_GroundingDINO scoring failed")
        return {"success": False, "error": str(exc)}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
