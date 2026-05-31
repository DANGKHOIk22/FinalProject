import io
import os
import base64
import logging
import requests
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class MediaResolver:
    @staticmethod
    def get_sas_url(material_item: str, media_map: dict = None) -> str:
        """
        Retrieves the SAS URL for a given image ID (UUID) from the provided media_map.
        If the item is already a URL, returns it directly.
        """
        if isinstance(material_item, str) and material_item.startswith("http"):
            return material_item

        actual_map = media_map or {}
        sas_url = actual_map.get(material_item)
        if sas_url:
            return sas_url

        raise FileNotFoundError(
            f"Could not resolve material item '{material_item}'. "
            f"It is not a valid local path, and was not found in the media_map."
        )

    @staticmethod
    def to_pil_image(material_item: str, media_map: dict = None) -> Image.Image:
        """
        Loads the media item as a PIL Image.
        Supports both local file paths and image IDs (UUIDs) resolved through media_map in RAM.
        """
        if isinstance(material_item, str) and os.path.exists(material_item):
            logger.info(f"Loading local PIL image: {material_item}")
            return Image.open(material_item).convert("RGB")

        sas_url = MediaResolver.get_sas_url(material_item, media_map)
        logger.info(f"Downloading PIL image from SAS URL for ID: {material_item}")
        response = requests.get(sas_url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    @staticmethod
    def to_opencv_image(material_item: str, media_map: dict = None) -> np.ndarray:
        """
        Loads the media item as an OpenCV image (numpy array).
        Supports both local file paths and image IDs (UUIDs) resolved through media_map in RAM.
        """
        if isinstance(material_item, str) and os.path.exists(material_item):
            logger.info(f"Loading local OpenCV image: {material_item}")
            return cv2.imread(material_item)

        sas_url = MediaResolver.get_sas_url(material_item, media_map)
        logger.info(f"Downloading OpenCV image from SAS URL for ID: {material_item}")
        response = requests.get(sas_url, timeout=30)
        response.raise_for_status()
        image_bytes = np.frombuffer(response.content, dtype=np.uint8)
        return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    @staticmethod
    def to_base64(material_item: str, media_map: dict = None) -> str:
        """
        Loads the media item and encodes it to a Base64 string in RAM.
        Supports both local file paths and image IDs (UUIDs) resolved through media_map in RAM.
        """
        if isinstance(material_item, str) and os.path.exists(material_item):
            logger.info(f"Loading and encoding local image to Base64: {material_item}")
            with open(material_item, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        sas_url = MediaResolver.get_sas_url(material_item, media_map)
        logger.info(f"Downloading and encoding image to Base64 from SAS URL for ID: {material_item}")
        response = requests.get(sas_url, timeout=30)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")

    @staticmethod
    def get_video_capture(material_item: str, media_map: dict = None) -> cv2.VideoCapture:
        """
        Returns a cv2.VideoCapture stream for the media item.
        Supports both local file paths and video SAS URLs resolved through media_map.
        """
        if isinstance(material_item, str) and os.path.exists(material_item):
            logger.info(f"Opening local video stream: {material_item}")
            return cv2.VideoCapture(material_item)

        sas_url = MediaResolver.get_sas_url(material_item, media_map)
        logger.info(f"Streaming video directly from Azure Storage SAS URL for ID: {material_item}")
        return cv2.VideoCapture(sas_url)
