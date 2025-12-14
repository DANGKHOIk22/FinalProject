import logging
import os
from datetime import datetime
from typing import Any, Type, Optional, List, Dict,Literal, Annotated

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
from langchain.tools import BaseTool, InjectedState
from langchain.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.config import DEFAULT_MODEL, MODEL_SEGMENT,SEGMENT_IMAGE_FOLDER
from app.prompts.toolbox.segment import get_segment_prompt_template

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)
# --- Input Schema ---
class SegmentInput(BaseModel):
    query_entity_recognition_task: Optional[str] = Field(
        default=None,
        description=(
            "The query is used to identify the information relevant to the entity."
            "Rule: When generating the final query, do NOT include any content related to:"
            "- Any content related to matches or games"
            "- Any content related to team names"
            "- Any content related to player names"
            "- Any content related to referee names"
            "- Any content related to stadium or venue names"
            "Only extract the core information requested by the user that is not tied to the above entities."
        ),
        examples=["the player in the red jersey", 
                  "the player wearing number 10",
                 ]
    )
    material: str = Field(..., description="Path to the image file")
class SplitEntityOutput(BaseModel):
    segments: List = Field(..., description="List of entity descriptions to be processed.") 

# --- Segment Tool ---
class SegmentTool(BaseTool):
    name: str = "segment"
    description: str = "A tool that returns the segmented region, which improves entity_recognition accuracy."
    args_schema: Type[BaseModel] = SegmentInput  
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    
    
    _llm: BaseChatModel = PrivateAttr()
    _processor: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)

    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__()
        self._llm = llm or ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL, 
            temperature=0.5,  
            top_p=0.95
        )
        self._load_models()
    def _load_models(self):
        """Lazy load heavy models only when needed."""
        if self._model is None or self._processor is None:
            self._device = infer_device()
            # Get Hugging Face token from environment if available
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            token_kwargs = {"token": hf_token} if hf_token else {}
            self._processor = AutoProcessor.from_pretrained(MODEL_SEGMENT, **token_kwargs)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_SEGMENT, **token_kwargs).to(self._device)
            logger.info("✅ Zero-Shot Detection Model loaded successfully.")
    
    @staticmethod
    def _prepare_batch_inputs(query_dict_tasks: dict) -> List:
        parser = PydanticOutputParser(pydantic_object=SplitEntityOutput)
        batch_inputs = []
        for query in query_dict_tasks.values():
            batch_inputs.append({
                "output_format": parser.get_format_instructions(),
                "query": query
            })
        return batch_inputs
    
    def _split_entities(self, query_dict_tasks: dict) -> List:
        """
        Split entities based on specific rules for soccer domain.

        Args:
            query_dict_tasks: Dictionary containing query tasks.
        Returns:
            A list of processed entity descriptions.
        """
         # Create output parser
        parser = PydanticOutputParser(pydantic_object=SplitEntityOutput)
        split_entity_prompt_template = get_segment_prompt_template()
        split_entity_chain = split_entity_prompt_template | self._llm | parser
        batch_inputs = self._prepare_batch_inputs(query_dict_tasks)
        try:
            response = split_entity_chain.batch(batch_inputs)
            logger.info(f"Segmented entities: {response}")
            return [res.segments for res in response]
        except Exception as e:
            logging.error(f"Failed to split entities: {str(e)}")
          
    def _detect_and_segment(self,image: Image.Image, entities_description: List[str]) -> List:
        """
        Get segmented entities from the query using the splitting logic.

        Args:
            image: Image object.
            entities_description: List of entity descriptions to be processed.
        Returns:
            A list of dictionaries containing segmented entity information.
        """
       
        images = [image] * len(entities_description) # Fix: in the future, there are more than one image inputs
        try:
            inputs = self._processor(images=images, text=entities_description, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.4,
                target_sizes=[image.size[::-1] for image in images]  
            )
            for result in results:
                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                labels = result["labels"] 
                result["boxes"] = boxes
                result["scores"] = scores
                result["labels"] = labels
            return results
        except Exception as e:
            error_msg = f"Failed to get segmented entities: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _run(
        self,
        query_entity_recognition_task: Optional[str] = None,
        material: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[str]:
        """
        Execute the segmentation tool.
        """
        run_tree = get_current_run_tree()
        
        try:
            # 1. Load Image
            if not os.path.exists(material):
                return f"Error: Image file not found at {material}"
                
            image = Image.open(material).convert("RGB")
            query_dict_tasks = {k: v for k, v in [("entity_recognition", query_entity_recognition_task)] if v is not None}
            tasks = list(query_dict_tasks.keys())
            # 2. Get Entities (LLM)
            entities_description = self._split_entities(query_dict_tasks) 
            logger.info(f"Entities to segment: {entities_description}")

            # 3. Detect Objects (Model)
            segmented_entities = self._detect_and_segment(image, entities_description)
            
        
            count = 0
            segmented_images = []
            for entity_idx,segmented_entity in enumerate(segmented_entities):
                for box, score, label in zip(segmented_entity["boxes"], segmented_entity["scores"], segmented_entity["labels"]):
                    x_min, y_min, x_max, y_max = box.tolist()

                    # Crop the object from the original image
                    segmented_object = image.crop((x_min, y_min, x_max, y_max))
                    segmented_images.append((segmented_object, label, score.item()))
                    
                    count += 1
            
                # 5. Save segmented objects to segmented_images folder
                original_filename = os.path.basename(material)
                name_without_ext = os.path.splitext(original_filename)[0]
                
                
                os.makedirs(SEGMENT_IMAGE_FOLDER, exist_ok=True)
                
                segmented_paths = []
                for idx, (segmented_img, label, score) in enumerate(segmented_images):
                    # Sanitize label for filename
                    safe_label = label.replace(" ", "_").replace("/", "-")
                    segmented_filename = f"{tasks[entity_idx]}_{name_without_ext}_{safe_label}_{idx+1}.png"
                    segmented_path = os.path.join(SEGMENT_IMAGE_FOLDER, segmented_filename)
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
