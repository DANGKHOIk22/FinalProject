import os
import logging
from deepface import DeepFace
import base64
import io
import json
from typing import Any, Dict
logger = logging.getLogger(__name__)


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one


    # DeepFace weights folder path registered in Azure. All model weights will be stored in .deepface/weights
    os.environ["DEEPFACE_HOME"] = os.getenv("AZUREML_MODEL_DIR", "")
    logger.info(f"DeepFace model path: {os.getenv('DEEPFACE_HOME')}")

    # Load model weights once during initialization. When call build_model, Deepface will look for weights in DEEPFACE_HOME path and create a singleton model instance
    global model_recognition_name, model_detector_name
    model_recognition_name = "Facenet512" # This is registered model on Azure ML workspace
    model_detector_name = "retinaface" # This is registered model on Azure ML workspace
    DeepFace.build_model(task="facial_recognition", model_name=model_recognition_name)
    DeepFace.build_model(task="face_detector", model_name=model_detector_name)
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

    def _base64_to_filelike(image_b64: str) -> io.BytesIO:
        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except Exception:
            # Some clients send base64 without padding; try a more forgiving decode
            padded = image_b64 + "=" * (-len(image_b64) % 4)
            image_bytes = base64.b64decode(padded)
        return io.BytesIO(image_bytes)
    
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

        return options


    def _json_safe(obj: Any) -> Any:
        # DeepFace outputs are usually JSON-friendly, but guard against numpy types.
        def _default(o: Any) -> Any:
            if hasattr(o, "tolist"):
                return o.tolist()
            if hasattr(o, "item"):
                return o.item()
            return str(o)

        return json.loads(json.dumps(obj, default=_default))

    logger.info("Run started")
    print("Run started")
    try:
        payload = _parse_payload(raw_data)
        image_b64 = _extract_base64_image(payload)
        image_file = _base64_to_filelike(image_b64)
        options = _extract_options(payload)

        embedding_objs = DeepFace.represent(
            img_path=image_file,
            model_name=model_recognition_name,
            detector_backend=model_detector_name,
            normalization="Facenet2018",
            enforce_detection=False,
            max_faces=options["max_faces"]
        )

        # Filter embeddings by confidence threshold
        embedding_objs = [
            obj for obj in embedding_objs
            if obj.get("face_confidence", 0.0) >= options["confidence_threshold"]
        ]
        print("Run completed successfully")
        logger.info("Run completed successfully")
        print(embedding_objs)
        logger.info(f"Embeddings: {embedding_objs}")
        
        return {
            "success": True,
            "result": _json_safe(embedding_objs),
            "metadata": {
                "model_recognition": model_recognition_name,
                "model_detector": model_detector_name,
                "max_faces": options["max_faces"],
                "confidence_threshold": options["confidence_threshold"]
            }
        }
    except Exception as e:
        logger.exception("Scoring failed")
        print(f"Scoring failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


