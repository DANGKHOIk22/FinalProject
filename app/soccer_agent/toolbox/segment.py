import logging
import os
import base64
import requests
import json
from datetime import datetime

from typing import Any, Type, Optional, List, Dict, Literal, Tuple
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr
from langsmith import get_current_run_tree
from langchain.tools import BaseTool
from langchain.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForToolRun
from app.config.config import SEGMENT_IMAGE_FOLDER
from app.config import settings
from app.soccer_agent.toolbox._config_loader import tool_description

# Setup logger
logger = logging.getLogger(__name__)
# --- Input Schema ---
class SegmentInput(BaseModel):
    query_entity_recognition_task: List[str] = Field(
        ...,
        description=(
            "Instruction: Analyze the user input and extract ONLY the text describing the visually identifiable object(s) that need to be located. Adhere to these strict rules:"
            "1. MANDATORY OBJECT CLASS: You MUST include the noun identifying the object type (e.g., 'person', 'ball', 'man', 'woman'). Never output an adjective without its noun (e.g., return 'a person in pink', NOT just 'pink'). Use 'person', 'man', 'woman' for humans, NOT 'player', 'athlete', or specific roles."
            "2. VISUAL ATTRIBUTES ONLY: Include color, clothing, and position (e.g., 'wearing a white shirt', 'on the left')."
            "3. REMOVE NAMED ENTITIES: Remove all proper names (e.g., 'Messi', 'Chelsea'). The segmentation tool does not recognize names, only descriptions."
            "4. REMOVE ABSTRACT CONTEXT: Remove all text related to actions, statistics, or comparisons (e.g., 'goals scored', 'compare', 'history')."
            "5. MULTIPLE OBJECTS: If multiple objects are described, separate them into distinct descriptions even if they are in a single sentence. Return each description as a separate item in the list."
            "6. LANGUAGE: Respond ONLY in English, regardless of the input language."),
        examples=[["the person wearing a white shirt on the left"], 
                  ["the person wearing number 10"],
                  ["the person in black uniform", "the person wearing green shirt"]
                 ]
    )
    material: List[str] = Field(..., description="Paths to the image files")

# --- Segment Tool ---
class SegmentTool(BaseTool):
    name: str = "segment"
    description: str = "Segments an image into cropped regions, each containing a detected human face. Use this when the agent needs to distinguish multiple people in a single image by visual attributes (e.g., clothing, color, face), which helps overcome the entity recognition tool's limitation of detecting all faces without differentiation. After segmentation, each cropped image can be passed to the entity recognition tool for per-face identification or attribute-based comparison. Returns the file paths of the cropped images as artifacts."
    args_schema: Type[BaseModel] = SegmentInput # type: ignore
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    
    
    _llm: BaseChatModel = PrivateAttr()
    _client: Any = PrivateAttr(default=None)

    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__(description=tool_description("segment"))
        self._initialize_endpoint()
        os.makedirs(SEGMENT_IMAGE_FOLDER, exist_ok=True)
    
    def _initialize_endpoint(self):
        """Initialize Qwen-VL endpoint via DashScope using OpenAI client."""
        from openai import OpenAI
        
        api_key = settings.DASHSCOPE_API_KEY
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY must be configured in settings")
        
        # Initialize OpenAI client with DashScope base URL
        # dashscope-intl.aliyuncs.com is the international endpoint recommended by Alibaba
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        
        logger.info("✅ Qwen-VL (DashScope International) client initialized via OpenAI SDK")
          
    def _detect_and_segment(self, image_path: str, entities_description: List[str]) -> List[Dict]:
        """
        Get segmented entities from the query using Qwen-VL.

        Args:
            image_path: Path to the image file.
            entities_description: Entity description to be processed.
        Returns:
            A list of dictionaries containing segmented entity information.
        """
        
        try:
            # 1. Get image dimensions and encode as base64
            from PIL import Image as PILImage
            with PILImage.open(image_path) as img:
                width, height = img.size
                
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            system_prompt = (
                """You are a helpful assistant to detect objects in images. 
                When asked to detect elements based on a description, 
                you return a valid JSON object containing bounding boxes for all elements in the form:
                `[{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "placeholder"}, ...]`. 
                For example, a valid response could be: 
                `[{"bbox_2d": [10, 30, 20, 60], "label": "placeholder"}, {"bbox_2d": [40, 15, 52, 27], "label": "placeholder"}]`.
                Return ONLY ONE bounding box for the single most prominent person matching the description.
                """
            )
            
            queries = ".".join(entities_description)
            user_prompt = (
                f"Detect ONLY ONE bounding box for the single most prominent person that best matches the description. "
                f"Do NOT return multiple boxes. Description: {queries}"
            )
            
            # Call Qwen-VL via OpenAI client
            response = self._client.chat.completions.create(
                model="qwen3-vl-flash-2026-01-22",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ],
                temperature=0.1,
                top_p=0.1,
                extra_headers={
                    "X-DashScope-WorkSpace": "" # Optional: specify workspace if needed
                }
            )

            # Extract response content
            try:
                content_str = response.choices[0].message.content
                # Clean markdown JSON blocks if present
                content_str = content_str.strip()
                if content_str.startswith("```json"):
                    content_str = content_str[7:]
                if content_str.startswith("```"):
                    content_str = content_str[3:]
                if content_str.endswith("```"):
                    content_str = content_str[:-3]
                
                detected_objects = json.loads(content_str.strip())
                logger.debug(f"Raw detected objects from Qwen-VL: {detected_objects}")
            except Exception as e:
                raw_content = response.choices[0].message.content if hasattr(response, 'choices') else "N/A"
                logger.error(f"Failed to extract or parse JSON from Qwen-VL response: {str(e)}\nRaw Response Content: {raw_content}")
                raise Exception(f"Failed to parse model output: {str(e)}")

            if not isinstance(detected_objects, list):
                logger.warning(f"Expected a list of detected objects, got {type(detected_objects)}. Attempting to wrap in list.")
                detected_objects = [detected_objects]
                
            results_data = []
            
            # Map Qwen-VL `bbox_2d` output to the expected schema
            for item in detected_objects:
                bbox = item.get("bbox_2d")
                if not bbox or len(bbox) != 4:
                    logger.warning(f"Skipping undefined bounding box: {item}")
                    continue
                    
                # 3. Scale normalized [0, 1000] coordinates to absolute pixel coordinates
                # Qwen-VL returns coordinates normalized to 1000
                x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox
                
                x_min = (x_min_norm / 1000.0) * width
                y_min = (y_min_norm / 1000.0) * height
                x_max = (x_max_norm / 1000.0) * width
                y_max = (y_max_norm / 1000.0) * height
                
                label = item.get("label", "extracted_object")
                
                results_data.append({
                    "box": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    },
                    "score": 1.0, # Dummy high score for Qwen-VL deterministic detections
                    "label": label
                })

            logger.info(f"✅ Qwen-VL response received and scaled successfully. Found {len(results_data)} objects")
            return results_data
            
        except Exception as e:
            error_msg = f"Failed to get segmented entities: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _extract_largest_face(self, pil_image: Image.Image, padding: float = 0.20) -> Image.Image:
        """
        Detect faces in the given PIL image using OpenCV Haar Cascade and return
        a padded crop of the single largest face found.
        If no face is detected, the original image is returned as-is (safe fallback).

        Args:
            pil_image:  PIL image (the coarse crop from Qwen-VL).
            padding:    Fraction of face size to add as padding on each side (default 20%).
        Returns:
            PIL image containing only the dominant face (or original crop if no face found).
        """
        # Convert PIL → OpenCV BGR → grayscale for detection
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Use the built-in frontal-face cascade (ships with opencv-python)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            logger.info("No face detected in crop — returning original coarse crop.")
            return pil_image

        # Pick the largest face by area (w * h)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        if len(faces) > 1:
            logger.info(f"Multiple faces found in crop ({len(faces)}) — keeping largest one.")

        # Add padding (clamped to image boundaries)
        img_w, img_h = pil_image.size
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        return pil_image.crop((x1, y1, x2, y2))

    def _post_proccessing_segmented_entities(self, image_path: str, segmented_entities: List[Dict]) -> List[str]:
        """
        Post-process and save segmented entities as image files.
        Args:
            image_path: Path to the original image file.
            segmented_entities: List of segmented entity dictionaries from detection.
        Returns:
            A list of file paths to the saved segmented images.
        """
        image = Image.open(image_path).convert("RGB")
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        segmented_paths = []
        for entity_idx, segmented_entity in enumerate(segmented_entities):
            box: Dict[str, float] = segmented_entity.get("box", {})
            x_min, y_min, x_max, y_max = box.values()
            label: str = segmented_entity.get("label", "")

            # Step 1: Coarse crop from Qwen-VL bounding box
            coarse_crop = image.crop((x_min, y_min, x_max, y_max))

            # Step 2: Refine to a single dominant face using OpenCV
            segmented_object = self._extract_largest_face(coarse_crop)

            # Save segmented objects to temporary/segmented_images folder
            safe_label = label.replace(" ", "_").replace("/", "-")
            segmented_filename = f"entity_recognition_{safe_label}_{timestamp}_{entity_idx+1}.png"
            segmented_path = os.path.join(SEGMENT_IMAGE_FOLDER, segmented_filename)
            segmented_paths.append(segmented_path)
            segmented_object.save(segmented_path)
            logger.info(f"✅ Cropped object saved to: {segmented_path}")
        return segmented_paths

    def _run(
        self,
        query_entity_recognition_task: List[str],
        material: List[str] = [],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[str]]:
        """
        Execute the segmentation tool.
        Returns tuple of (status_message, segmented_paths_string)
        """
        run_tree = get_current_run_tree()
        
        try:
            # 1. Load Image
            image_path = material[0]  # TODO: fix to support multiple images
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Material file not found: {image_path}")
            logger.info(f"✅ Image loaded successfully from: {image_path}")
            # 2. Detect Objects (Model)
            segmented_entities = self._detect_and_segment(image_path=image_path, entities_description=query_entity_recognition_task)
            
            # 3. Post-process and Save Segmented Objects
            segmented_paths = self._post_proccessing_segmented_entities(image_path=image_path, segmented_entities=segmented_entities)
            
            return (
                f"Successfully segmented objects. The tool segmented the image into: {len(segmented_paths)} parts. "
                f"Segmented image paths: {', '.join(segmented_paths)}",
                segmented_paths,
            )

        except Exception as e:
            error_msg = f"Error in segment_tool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if run_tree:
                run_tree.end(error=error_msg)

            # Return detailed error message to the Agent
            return "An error occurred while segmenting the image. Try rephrase the tool input or stop the execution. Error: {error_msg}", []
