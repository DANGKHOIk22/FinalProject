import os
import sys
import logging
import json
import base64
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    global logger
    logger = logging.getLogger()
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(stream_handler)

    global app
    model_path = Path(os.getenv("AZUREML_MODEL_DIR", "")) / ".insightface"
    app = FaceAnalysis(name="buffalo_l", root=str(model_path), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
    logger.info(f"InsightFace model path: {model_path}")
    logger.info("Init complete")

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    
    :param raw_data: The raw JSON data passed to the endpoint. It is expected to be a JSON object with a base64-encoded image string under the "image" key and optional parameters. These options can include: max_faces (int), confidence_threshold (float).
    :return: A JSON-serializable dictionary with the scoring results or error information.
    """
    def _parse_payload(data: Any) -> Dict[str, Any]:
        if isinstance(data, dict):
            return data
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8", errors="replace")
        if isinstance(data, str):
            data = data.strip()
            if not data:
                return {}
            return json.loads(data)
        # Fallback for AzureML SDKs that sometimes pass already-decoded objects
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
        # Allow data URLs: data:image/jpeg;base64,<...>
        if image_value.startswith("data:") and "," in image_value:
            image_value = image_value.split(",", 1)[1].strip()
        if not image_value:
            raise ValueError('Field "image" is empty')
        return image_value

    def _base64_to_filelike(image_b64: str) -> np.ndarray:
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            # Some clients send base64 without padding; try a more forgiving decode
            padded = image_b64 + "=" * (-len(image_b64) % 4)
            image_bytes = base64.b64decode(padded)

        # Decode directly into an OpenCV BGR image to avoid an intermediate BytesIO wrapper.
        img_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError('Field "image" does not contain a valid decodable image')
        return image_bgr
    
    def _pad_image(image: np.ndarray, dest_size: tuple = (640, 640)) -> np.ndarray:
        """
        Pad the image to the destination size if it's smaller.
        Centers the original image in the padded output with black borders.
        
        :param image: Input BGR image as numpy array
        :param dest_size: Target size as (height, width)
        :return: Padded image with dest_size dimensions
        """
        dest_height, dest_width = dest_size
        height, width = image.shape[:2]
        
        # Calculate padding needed
        pad_height = dest_height - height if height < dest_height else 0
        pad_width = dest_width - width if width < dest_width else 0
        
        if pad_height == 0 and pad_width == 0:
            return image
        
        # Center the image by distributing padding evenly
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        
        logger.info(f"Padding image: original {width}x{height}, padding top={top} bottom={bottom} left={left} right={right}")
        
        image_padded = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        logger.info(f"Original image dimensions: {width}x{height}")
        logger.info(f"Padded image dimensions: {image_padded.shape[1]}x{image_padded.shape[0]}")
        
        return image_padded
    
    def _extract_options(payload: Dict[str, Any]) -> Dict[str, Any]:
        options = {}
        # Extract "max_faces"
        options["max_faces"] = 10
        if "max_faces" in payload:
            max_faces = int(payload["max_faces"])

            # Validate max_faces
            if 0 < max_faces and max_faces <= 10:
                options["max_faces"] = max_faces

        # Extract "confidence_threshold"
        options["confidence_threshold"] = 0.85
        if "confidence_threshold" in payload:
            confidence_threshold = float(payload["confidence_threshold"])

            # Validate confidence_threshold
            if 0.0 < confidence_threshold and confidence_threshold < 1.0:
                options["confidence_threshold"] = confidence_threshold

        logger.info(f"📋 Options - max_faces: {options['max_faces']}, confidence_threshold: {options['confidence_threshold']:.3f}")
        return options

    def _face_to_dict(face: Any) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if hasattr(face, "bbox") and face.bbox is not None:
            data["bbox"] = face.bbox.tolist() if hasattr(face.bbox, "tolist") else list(face.bbox)
        if hasattr(face, "det_score") and face.det_score is not None:
            data["det_score"] = float(face.det_score)
        if hasattr(face, "kps") and face.kps is not None:
            data["kps"] = face.kps.tolist() if hasattr(face.kps, "tolist") else face.kps
        if hasattr(face, "embedding") and face.embedding is not None:
            data["embedding"] = face.embedding.tolist() if hasattr(face.embedding, "tolist") else face.embedding
        if hasattr(face, "gender") and face.gender is not None:
            data["gender"] = int(face.gender)
        if hasattr(face, "age") and face.age is not None:
            data["age"] = int(face.age)
        return data

    def _json_safe(obj: Any) -> Any:
        def _default(o: Any) -> Any:
            if hasattr(o, "tolist"):
                return o.tolist()
            if hasattr(o, "item"):
                return o.item()
            return str(o)

        return json.loads(json.dumps(obj, default=_default))

    logger.info("Run started")
    try:
        payload = _parse_payload(raw_data)
        image_b64 = _extract_base64_image(payload)
        image_file = _base64_to_filelike(image_b64)
        image_file = _pad_image(image_file, dest_size=(640, 640))
        options = _extract_options(payload)

        faces = app.get(image_file)
        logger.info(f"Total faces detected before filtering: {len(faces)}")
        logger.info("Rawfaces: %s", [_face_to_dict(face) for face in faces])
        pre_filter_scores = [float(getattr(face, "det_score", 0.0)) for face in faces]
        logger.info(
            "Det scores before filtering: %s",
            pre_filter_scores,
        )
        
        # Keep only high-confidence faces and cap by max_faces.
        faces = [
            face for face in faces
            if float(getattr(face, "det_score", 0.0)) >= options["confidence_threshold"]
        ]
        post_filter_scores = [float(getattr(face, "det_score", 0.0)) for face in faces]
        logger.info(
            "Faces after confidence filtering: count=%d threshold=%.3f det_scores=%s",
            len(faces),
            options["confidence_threshold"],
            post_filter_scores,
        )
        faces = sorted(faces, key=lambda x: float(getattr(x, "det_score", 0.0)), reverse=True)
        faces = faces[: options["max_faces"]]

        logger.info(
            "Faces after max_faces filtering: count=%d max_faces=%d",
            len(faces),
            options["max_faces"],
        )
        result = [_face_to_dict(face) for face in faces]
        logger.info("Run completed successfully")
        logger.info(f"Detected faces: {len(result)}")
        return {
            "success": True,
            "result": _json_safe(result),
            "metadata": {
                "model": "buffalo_l",
                "provider": "CPUExecutionProvider",
                "max_faces": options["max_faces"],
                "confidence_threshold": options["confidence_threshold"],
            },
        }

    except Exception as e:
        logger.exception("Scoring failed")
        return {
            "success": False,
            "error": str(e),
        }