import base64
import io
import json
import logging
import os
from typing import Any, Dict, List
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logger = logging.getLogger(__name__)


def init():
    """Load GroundingDINO model and processor once per container."""
    model_dir = os.getenv("AZUREML_MODEL_DIR", "")
    if not model_dir:
        raise EnvironmentError("AZUREML_MODEL_DIR is not set")

    base_model_dir = Path(model_dir)
    model_dir_path = base_model_dir / ".groundingdino" if (base_model_dir / ".groundingdino").exists() else base_model_dir
    print(model_dir_path)

    global device, processor, model
    device = Accelerator().device
    processor = AutoProcessor.from_pretrained(model_dir_path)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir_path).to(device)
    model.eval()
    logger.info("GroundingDINO initialized on device %s", device)


def run(raw_data):
    """Perform zero-shot object detection with GroundingDINO.
    
    Args:
        raw_data: The input payload containing the base64-encoded image and other parameters. It should be a JSON dictionary with at least the "image" field and "queries" field.
    Returns:
        
    """

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

    def _extract_base64_image(payload: Dict[str, Any]) -> str:
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

    def _json_safe(obj: Any) -> Any:
        def _default(o: Any) -> Any:
            if hasattr(o, "tolist"):
                return o.tolist()
            if hasattr(o, "item"):
                return o.item()
            return str(o)

        return json.loads(json.dumps(obj, default=_default))

    def _pil_to_base64(img: Image.Image) -> str:
        buff = io.BytesIO()
        img.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    try:
        payload = _parse_payload(raw_data)
        if "processor" not in globals() or "model" not in globals():
            init()

        # Extract and decode image
        image_b64 = _extract_base64_image(payload)
        image = _base64_to_pil(image_b64)

        # Extract text queries
        text_queries = payload.get("queries") or payload.get("text")
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        if not text_queries or not isinstance(text_queries, list):
            raise ValueError('Field "queries" must be a non-empty list of strings')

        # Extract other parameters
        threshold = float(payload.get("threshold", 0.6))
        text_threshold = float(payload.get("text_threshold", 0.6))
        return_crops = bool(payload.get("return_crops", False))
        max_detections = payload.get("max_detections", 10)
        if max_detections is not None:
            max_detections = int(max_detections)

        images: List[Image.Image] = [image]
        inputs = processor(images=images, text=text_queries, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
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
            "detections": _json_safe(detections),
            "meta": {
                "threshold": threshold,
                "text_threshold": text_threshold,
                "num_queries": len(text_queries),
                "device": str(device),
            },
        }
    except Exception as e:
        logger.exception("GroundingDINO scoring failed")
        return {
            "success": False,
            "error": str(e),
        }