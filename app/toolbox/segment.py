import logging
import os
from datetime import datetime
from typing import Any, Type, Optional, List, Dict,Literal

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, PrivateAttr

import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    infer_device,
)

from langsmith import get_current_run_tree
from langchain.tools import BaseTool
from langchain.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.config import DEFAULT_MODEL, MODEL_SEGMENT
from app.prompts.toolbox.segment import get_segment_prompt_template

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)
# --- Input Schema ---
class SegmentInput(BaseModel):
    query: str = Field(
        ..., 
        description=(
            "The specific visual description of the entity to be localized within the image. "
            "RULES: 1. Extract ONLY the visual noun phrase (e.g., 'player in white', 'the ball'). "
            "2. REMOVE procedural commands (e.g., 'find', 'compare', 'count', 'analyze'). "
            "3. If the user asks to compare X and Y, provide the description of the entity that needs segmentation."
        ),
        examples=["player in the red jersey", "referee holding a card", "goalkeeper"]
    )
    material: str = Field(..., description="Path to the image file")
class SplitEntityOutput(BaseModel):
    segments: List = Field(..., description="List of entity descriptions to be processed.") 
    
# --- Segment Tool ---
class SegmentTool(BaseTool):
    name: str = "segment"
    description: str = "A tool that segments objects in images based on textual descriptions. It takes a text description and an image file path as input and returns the segmented object from the image."
    args_schema: Type[BaseModel] = SegmentInput  
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    
    
    _llm: BaseChatModel = PrivateAttr()
    _processor: Any = PrivateAttr() 
    _model: Any = PrivateAttr()

    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__()
        self._llm = llm or ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL, 
            temperature=0.5,  
            top_p=0.95
        )
        device = infer_device()
        self._processor = AutoProcessor.from_pretrained(MODEL_SEGMENT)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_SEGMENT).to(device)
    def _split_entities(self, query: str) -> List:
        """
        Split entities based on specific rules for soccer domain.

        Args:
            entities: List of entity dictionaries to be processed.
        Returns:
            A list of processed entity descriptions.
        """
         # Create output parser
        parser = PydanticOutputParser(pydantic_object=SplitEntityOutput)
        split_entity_prompt_template = get_segment_prompt_template()
        split_entity_chain = split_entity_prompt_template | self._llm | parser
        
        try:
            response = split_entity_chain.invoke({
                "output_format": parser.get_format_instructions(),
                "query": query
            })
            return response.segments
        except Exception as e:
            error_msg = f"Failed to split entities from query '{query}': {str(e)}"
            logging.error(error_msg)
            # Raise with context
            raise RuntimeError(error_msg) from e
    def _get_segmented_entities(self,image_path:str, entities_description: List[str]) -> Dict:
        """
        Get segmented entities from the query using the splitting logic.

        Args:
            image_path: Path to the image file.
            entities_description: List of entity descriptions to be processed.
        Returns:
            A list of dictionaries containing segmented entity information.
        """
        image = Image.open(image_path).convert("RGB") 
        try:
            inputs = self._processor(images=image, text=[entities_description], return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )
            return results[0]
        except Exception as e:
            error_msg = f"Failed to get segmented entities from image '{image_path}': {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    
    def _run(self, query: str, material: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[str]:
        """
        Execute the segmentation tool.
        """
        run_tree = get_current_run_tree()
        
        try:
            # 1. Load Image
            if not os.path.exists(material):
                return f"Error: Image file not found at {material}"
                
            image = Image.open(material).convert("RGB")
            
            # 2. Get Entities (LLM)
            entities_description = self._split_entities(query) 
            logger.info(f"Entities to segment: {entities_description}")

            # 3. Detect Objects (Model)
            segmented_entities = self._get_segmented_entities(material, entities_description)
            
        
            count = 0
            segmented_images = []
            for box, score, label in zip(segmented_entities["boxes"], segmented_entities["scores"], segmented_entities["labels"]):
                x_min, y_min, x_max, y_max = box.tolist()

                # Crop the object from the original image
                segmented_object = image.crop((x_min, y_min, x_max, y_max))
                segmented_images.append((segmented_object, label, score.item()))
                
                count += 1
            
            # 5. Save segmented objects to segmented_images folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = os.path.basename(material)
            name_without_ext = os.path.splitext(original_filename)[0]
            
            
            segmented_folder = "segmented_images"
            os.makedirs(segmented_folder, exist_ok=True)
            
            segmented_paths = []
            for idx, (segmented_img, label, score) in enumerate(segmented_images):
                # Sanitize label for filename
                safe_label = label.replace(" ", "_").replace("/", "-")
                segmented_filename = f"{name_without_ext}_{safe_label}_{idx+1}_{timestamp}.png"
                segmented_path = os.path.join(segmented_folder, segmented_filename)
                segmented_img.save(segmented_path)
                segmented_paths.append(segmented_path)
                logger.info(f"✅ Cropped object saved to: {segmented_path}")
            segmented_paths = ", ".join(segmented_paths)
            return  "Successfully segmented objects.", segmented_paths

        except Exception as e:
            error_msg = f"Error in segment_tool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if run_tree:
                run_tree.end(error=error_msg)

            # Return detailed error message to the Agent
            return "An error occurred while segmenting the image. Try calling this tool again or stop the execution."


if __name__ == "__main__":
    """Test the SegmentTool with a sample image."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the tool
    print("Initializing SegmentTool...")
    tool = SegmentTool()
    
    # Test parameters
    test_image_path = "D:\\Project\\FinalProject\\example_images\\fc_womens_rank-11_24.jpeg"  # Replace with your test image path
    test_query = "a player is wearing a blue jersey and a player is with a ball"  # Replace with your test query

    
    print(f"\nTesting segmentation on: {test_image_path}")
    print(f"Query: {test_query}")
    print("-" * 50)
    
    # Run the tool
    result = tool._run(query=test_query, material=test_image_path)
    
    print("\nResult:")
    print(result)
