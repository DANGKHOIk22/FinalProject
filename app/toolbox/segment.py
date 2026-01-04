import logging
import os
import base64
import requests
from datetime import datetime

from typing import Any, Type, Optional, List, Dict, Literal, Tuple
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr
from langsmith import get_current_run_tree
from langchain.tools import BaseTool
from langchain.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForToolRun
from app.config.config import SEGMENT_IMAGE_FOLDER
from app.config import settings

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
    description: str = "A tool that returns the segmented region, which improves entity_recognition accuracy."
    args_schema: Type[BaseModel] = SegmentInput # type: ignore
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    
    
    _llm: BaseChatModel = PrivateAttr()
    _cg_endpoint_uri: Optional[str] = PrivateAttr(default=None)
    _cg_endpoint_key: Optional[str] = PrivateAttr(default=None)
    _cg_payload_header: Dict = PrivateAttr(default_factory=dict)

    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__()
        self._initialize_endpoint()
        os.makedirs(SEGMENT_IMAGE_FOLDER, exist_ok=True)
    
    def _initialize_endpoint(self):
        """Initialize GroundingDino endpoint connection."""
        # Prefer combined CLIP+GroundingDINO endpoint; fallback to legacy GD endpoint
        self._cg_endpoint_uri = (
            settings.CLIP_GROUNDINGDINO_ENDPOINT_URI
            or settings.GROUNDINGDINO_ENDPOINT_URI
        )
        self._cg_endpoint_key = (
            settings.CLIP_GROUNDINGDINO_ENDPOINT_KEY
            or settings.GROUNDINGDINO_ENDPOINT_KEY
        )
        
        if not self._cg_endpoint_uri or not self._cg_endpoint_key:
            raise ValueError("GroundingDINO/CLIP endpoint URI and key must be configured")
        
        self._cg_payload_header = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._cg_endpoint_key}'
        }
        
        # Test endpoint connectivity
        try:
            response = requests.post(
                url=self._cg_endpoint_uri,
                headers=self._cg_payload_header,
                json={"task": "groundingdino", "image": "", "queries": []},
                timeout=60
            )
            # We expect an error for empty payload, but 200/400 means endpoint is reachable
            if response.status_code in [200, 400]:
                logger.info("✅ GroundingDino endpoint is reachable")
            else:
                logger.warning(f"⚠️  GroundingDino endpoint returned unexpected status: {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to GroundingDino endpoint: {str(e)}")
          
    def _detect_and_segment(self, image_path: str, entities_description: List[str]) -> List[Dict]:
        """
        Get segmented entities from the query using Azure GroundingDino endpoint.

        Args:
            image_path: Path to the image file.
            entities_description: Entity description to be processed.
        Returns:
            A list of dictionaries containing segmented entity information.
        """
        
        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            # Prepare payload
            payload = {
                "task": "groundingdino",
                "image": image_base64,
                "text_threshold": 0.4,
                "threshold": 0.4,
                "queries": [(".").join(entities_description)]  # Group all queries together
            }
            
            # Call GroundingDino endpoint
            if not self._cg_endpoint_uri:
                raise ValueError("GroundingDino endpoint URI not configured")
            
            response = requests.post(
                url=self._cg_endpoint_uri,
                headers=self._cg_payload_header,
                json=payload,
                timeout=60
            )

            # Handle response
            if response.status_code != 200:
                raise Exception(f"GroundingDino endpoint request failed: {response.status_code} - {response.text}")
            
            try:
                response_dict = response.json()
            except Exception:
                raise Exception(f"Failed to parse endpoint response as JSON: {response.text}")
            
            if not response_dict.get("success", False):
                raise Exception(f"GroundingDino endpoint error: {response_dict.get('error', 'Unknown error')}")
            logger.info(response_dict)
            results_data = response_dict.get("detections", [])
            logger.info(f"✅ GroundingDino endpoint response received successfully. Found {len(results_data)} query groups")
            
            # results_data is a list of dictionaries, each corresponding to a detected object
            # Each dictionary contains keys: "box", "score", "label"
            # "box" is a dictionary for single detection with keys: x_min, y_min, x_max, y_max
            # "score" is a float for single detection
            # "label" is a predicted label string which is from the text query
            
            return results_data
            
        except Exception as e:
            error_msg = f"Failed to get segmented entities: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

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
            score = segmented_entity.get("score")
            label: str = segmented_entity.get("label", "")
            
            # Crop the object from the original image
            segmented_object = image.crop((x_min, y_min, x_max, y_max))            
        
            #Save segmented objects to temporary/segmented_images folder
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
                f"Successfully segmented objects. The tool found {len(segmented_paths)} entities. "
                f"Segmented image paths: {', '.join(segmented_paths)}",
                segmented_paths,
            )

        except Exception as e:
            error_msg = f"Error in segment_tool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if run_tree:
                run_tree.end(error=error_msg)

            # Return detailed error message to the Agent
            return "An error occurred while segmenting the image. Try calling this tool again or stop the execution.", []
