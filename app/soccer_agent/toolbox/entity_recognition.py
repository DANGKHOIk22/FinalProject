import json
import os
import logging
import base64
import pymongo
import numpy as np
import cv2
from collections import defaultdict
import requests

from dns import resolver
from langsmith import get_current_run_tree
from pymongo.server_api import ServerApi
from typing import Any, Tuple, Type, Optional, Literal, List, Dict
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from qdrant_client import QdrantClient, models
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from app.config.settings import settings
from app.config.config import QDRANT_SEARCH_SCORE_THRESHOLD as THRESHOLD
from app.schema.textual_entity_search import SearchingResult
from app.schema.soccerwiki_entities import PlayerSchema, RefereeSchema, VenueSchema, TeamSchema
from app.cache.standard_cache import standard_cache
from app.soccer_agent.toolbox._config_loader import tool_description
from langgraph.prebuilt import ToolRuntime


# Setup logger
logger = logging.getLogger(__name__)

class EntityRecognitionInput(BaseModel):
    image_id: str = Field(..., description="UUID of the image  with entity_recognition names; if omitted, defaults to all image paths.")

class EntityRecognitionTool(BaseTool):
    """
    Tool for recognizing soccer entities from images using face recognition.
    """
    
    name: str = "entity_recognition"
    description: str = """Given an image path, the tool retrieves the requiring entities of the question, and return its according WikiPage. You should use segment tool before this tool. The entity database contains the history and background knowledge for all the players, teams, venues, coaches and referees from games are from 2022 World Cup and 6 European major leagues (England Premier, Germany Bundesliga, Italy Serie-a, Spain Laliga, France Ligue-1 and European Champions League) during 2017-2024. Because this tool can retrieve detailed information about the entities, it's not necessary to call entity_augment after calling this tool."""
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    args_schema: Type[BaseModel] = EntityRecognitionInput # type: ignore
    
    # Internal state (not exposed to LLM)
    _qdrant_client: Optional[QdrantClient] = PrivateAttr(default=None)
    _mongo_client: Optional[pymongo.MongoClient] = PrivateAttr(default=None)
    _deepface_rep: Any = PrivateAttr(default=None)

    # Azure InsightFace Endpoint attributes
    _insight_endpoint_uri: str = PrivateAttr(default="")
    _insight_payload_header: Dict[str, str] = PrivateAttr(default={})
    _insight_endpoint_key: str = PrivateAttr(default="")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(description=tool_description("entity_recognition"), **data)
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Qdrant and MongoDB clients - use preloaded from main.py if available."""
        # Try to use preloaded clients from main.py
        try:
            import main
            if main.qdrant_client is not None:
                self._qdrant_client = main.qdrant_client
                logger.info("✅ Using preloaded Qdrant client from lifespan")
            if main.mongo_client is not None:
                self._mongo_client = main.mongo_client
                logger.info("✅ Using preloaded MongoDB client from lifespan")
            
        except (ImportError, AttributeError):
            pass
        
        # Fallback: Initialize clients if not preloaded
        # Initialize Qdrant client
        if self._qdrant_client is None:
            logger.info("Loading Qdrant client on demand...")
            qdrant_url = settings.QDRANT_URL
            qdrant_api_key = settings.QDRANT_API_KEY
            
            self._qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                prefer_grpc=True,
                check_compatibility=False,
                timeout=20
            )
            logger.info("✅ Qdrant client initialized successfully")
        
        # Initialize MongoDB client
        if self._mongo_client is None:
            logger.info("Loading MongoDB client on demand...")
            mongo_srv = settings.MONGO_SRV
            if mongo_srv:
                resolver.default_resolver = resolver.Resolver(configure=False)
                resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']
                self._mongo_client = pymongo.MongoClient(mongo_srv, server_api=ServerApi('1'))
                logger.info("✅ MongoDB client initialized successfully")
        
        # Intialize Azure Endpoint client
        self._insight_endpoint_uri = settings.INSIGHTFACE_ENDPOINT_URI or ""
        self._insight_endpoint_key = settings.INSIGHTFACE_ENDPOINT_KEY or ""
        
        # Validate endpoint configuration
        if not self._insight_endpoint_uri:
            logger.error(f"INSIGHTFACE_ENDPOINT_URI is empty or not configured. Value: '{self._insight_endpoint_uri}'")
            raise ValueError("INSIGHTFACE_ENDPOINT_URI is not configured. Please check settings.")
        if not self._insight_endpoint_key:
            logger.error("INSIGHTFACE_ENDPOINT_KEY is empty or not configured.")
            raise ValueError("INSIGHTFACE_ENDPOINT_KEY is not configured. Please check settings.")
        
        logger.info(f"InsightFace Endpoint URI: {self._insight_endpoint_uri}")
        
        self._insight_payload_header = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._insight_endpoint_key}'
        }
        response = requests.post(url=self._insight_endpoint_uri, headers=self._insight_payload_header, timeout=60)
        if response.status_code == 200:
            logger.info("✅ InsightFace endpoint is reachable")
        else:
            raise ConnectionError(f"Failed to connect to InsightFace endpoint: {response.status_code} - {response.text}")

    
    def _create_payload(self, image_id: str, max_faces: int = 5, confidence_threshold: float = 0.60, media_registry: Any = None, user_id: str = "default_user", thread_id: str = "default_thread") -> Dict[str, Any]:
        """
        Create payload for InsightFace endpoint.
        
        :param image_id: UUID to the image file
        :type image_id: str
        :param max_faces: Maximum number of faces to detect
        :type max_faces: int
        :param confidence_threshold: Confidence threshold for face detection
        :type confidence_threshold: float
        :param media_registry: MediaRegistryService instance
        :return: JSON payload for the request
        :rtype: Dict
        """
        try:
            # 1. Load image to Base64 in RAM using MediaRegistryService
            image_base64 = media_registry.get_base_64(user_id, thread_id, image_id)
            if not image_base64:
                raise ValueError(f"Could not load image {image_id}")
            logger.info(f"📸 Image loaded and Base64 encoded: {image_id}")
            
            # 2. Get OpenCV image to validate size
            pil_img = media_registry.get_pil_image(user_id, thread_id, image_id)
            if pil_img is None:
                error_msg = f"Invalid or corrupted image: {image_id}. Could not decode image."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            img_width, img_height = pil_img.size
            logger.info(f"✅ Image validated - Format: valid, Size: {img_width}x{img_height}")
        
        except Exception as e:
            error_msg = f"Failed to process image {image_id}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Validate confidence threshold
        if not (0.0 < confidence_threshold < 1.0):
            logger.warning(f"Invalid confidence_threshold {confidence_threshold}; using 0.6")
            confidence_threshold = 0.60
        
        payload = {
            "image": image_base64,
            "max_faces": max_faces,
            "confidence_threshold": confidence_threshold
        }
        
        logger.info(f"🔍 Payload created - max_faces: {max_faces}, confidence_threshold: {confidence_threshold}")
        return payload
    
    @staticmethod    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _extract_entities_from_image(self, image_id: str, media_registry: Any, user_id: str, thread_id: str) -> List[Dict]:
        """
        Extract soccer entities from image using face recognition.
        
        Args:
            image_id: Image UUID
            media_registry: MediaRegistryService instance
            
        Returns:
            List of entity dictionaries with ENTITY_TYPE and NAME
        """
        collection_name = settings.QDRANT_COLLECTION_NAME 
        assert collection_name is not None, "Qdrant client is not initialized"
        assert self._qdrant_client is not None, "Qdrant client is not initialized"
        
        # Create payload for InsightFace endpoint
        payload = self._create_payload(image_id=image_id, max_faces=5, confidence_threshold=0.60, media_registry=media_registry, user_id=user_id, thread_id=thread_id)

        try:
        # Call to InsightFace endpoint
            response: requests.Response = requests.post(
                url=self._insight_endpoint_uri,
                headers=self._insight_payload_header,
                json=payload,
                timeout=60
            )
        except requests.RequestException as e:
            raise Exception(f"InsightFace endpoint request error: {str(e)}")
        
        response_dict: Dict = response.json()
        # Handle response
        if response.status_code != 200:
            raise Exception(f"InsightFace endpoint request failed: {response.status_code} - {response.text}")
        if response_dict.get("success", False) is False:
            raise Exception(f"InsightFace endpoint error: {response_dict.get('error', 'Unknown error')} ")

        results = response_dict.get("result") or []
        logger.info(f"✅ InsightFace endpoint response received successfully. There are {len(results)} results")
        # Print out each result for debugging
        for idx, res in enumerate(results):
            logger.info(f"Result {idx+1}: bbox={res.get('bbox', [])}, Confidence: {res.get('det_score', 0.0)}")
        
        # Search entities in Qdrant
        detected_faces = results
        soccer_entities = []
        for idx, detected_face in enumerate(detected_faces):
            detected_face_embedding = detected_face.get("embedding", [])
            if not detected_face_embedding:
                logger.info("Face %d has empty embedding; skipping", idx + 1)
                continue

            search_result = self._qdrant_client.query_points(
                collection_name=collection_name,
                query=detected_face_embedding,
                limit=7,
                score_threshold=0.5,
                with_vectors=True
            )
            
            # Dictionary storing information for each entity
            candidates = defaultdict(lambda: {"ENTITY_TYPE": None, "max_score": 0, "count": 0, "score_list": []})
            
            # Voting
            for point in search_result.points:
                entity_name = point.payload['NAME']
                point_vectors = point.vector
                if not point_vectors:
                    continue
                match_count = 0
                for sub_vector in point_vectors:
                    score = self.cosine_similarity(detected_face_embedding, sub_vector)
                    if score >= THRESHOLD:
                        candidates[entity_name]["score_list"].append(score)
                        match_count += 1
                # Update candidate info
                candidates[entity_name]["count"] = match_count
                candidates[entity_name]["ENTITY_TYPE"] = point.payload.get("ENTITY_TYPE")
                candidates[entity_name]["max_score"] = point.score
            
            # Final Score (Re-ranking)
            ranked_candidates = []
            for entity_name, data in candidates.items():
                final_score = (0.55 * data["max_score"]) + (0.45 * data["count"] / 20) + (0.1 * np.mean(data["score_list"]))
                ranked_candidates.append((entity_name, final_score, data["max_score"], data["count"]))
            
            # Sort in descending order by Final Score
            ranked_candidates.sort(key=lambda x: x[1], reverse=True)
            
            if not ranked_candidates:
                raise ValueError("Can't find any matching entities in the database.")
            
            if ranked_candidates[0]:
                soccer_entities.append({
                    "ENTITY_TYPE": candidates[ranked_candidates[0][0]]["ENTITY_TYPE"],
                    "NAME": ranked_candidates[0][0]
                })
                logger.info(f"Face {idx+1}: Matched to {ranked_candidates[0]}")
        found_entity_names = [f"{e['NAME']} ({e['ENTITY_TYPE']})" for e in soccer_entities]
        logger.info(f"Using voting and re-ranking, recognized: {', '.join(found_entity_names)}")
        return soccer_entities
    
    def _parse_entity_result(self, entity_data: dict) -> Optional[PlayerSchema | RefereeSchema | VenueSchema | TeamSchema]:
        """
        Parse MongoDB result to appropriate Pydantic model based on entity type.
        
        Args:
            entity_data (dict): Raw entity data from MongoDB
            
        Returns:
            Optional[BaseModel]: Parsed Pydantic model instance or None if parsing fails
        """
        entity_type = entity_data.get('ENTITY_TYPE')
        
        if not entity_type:
            logger.warning(f"No ENTITY_TYPE found in data")
            return None
        
        schema_map: Dict[str, Type[PlayerSchema | RefereeSchema | VenueSchema | TeamSchema]] = {
            'venue': VenueSchema,
            'player': PlayerSchema,
            'team': TeamSchema,
            'referee': RefereeSchema
        }
        
        schema_class = schema_map.get(entity_type)
        if schema_class:
            try:
                return schema_class(**entity_data)
            except Exception as e:
                logger.warning(f"Failed to parse {entity_type}: {e}")
                return None
        
        logger.warning(f"Unknown entity type: {entity_type}")
        return None
    
    @standard_cache.cache(ttl=60 * 60, validatedModel=SearchingResult)
    def _query_database(self, soccer_entities: List[Dict]) -> SearchingResult:
        """
        Query MongoDB database for detailed information about entities.
        
        Args:
            soccer_entities: List of entities from Qdrant
            
        Returns:
            SearchingResult with found and missing entities
        """
        result = SearchingResult()
        
        if not self._mongo_client:
            logger.error("MongoDB client not initialized")
            return result
        
        try:
            database_name = settings.SOCCER_DB_NAME
            collection_name = settings.SOCCER_COLLECTION_NAME
            
            db = self._mongo_client.get_database(name=database_name)
            collection = db.get_collection(name=collection_name)
            
            # Query for each entity
            for entity in soccer_entities:
                found_entity: Optional[PlayerSchema | RefereeSchema | VenueSchema | TeamSchema] = None
                missing_entity: Optional[str] = None
                
                entity_type = entity['ENTITY_TYPE']
                name = entity['NAME']
                
                logger.info(f"Searching database for {entity_type}: {name}")
                
                # Build query filter
                filter_query = {
                    "$and": [
                        {"ENTITY_TYPE": entity_type},
                        {"NAME": name}
                    ]
                }
                
                try:
                    entity_data = collection.find_one(filter_query)
                    
                    if entity_data:
                        # Convert _id to string
                        if '_id' in entity_data:
                            entity_data['_id'] = str(entity_data['_id'])
                        
                        # Parse to Pydantic model
                        parsed_entity = self._parse_entity_result(entity_data)
                        if parsed_entity:
                            found_entity = parsed_entity
                            logger.info(f"✅ Found and parsed {entity_type}: {name}")
                        else:
                            logger.warning(f"⚠️  Failed to parse {entity_type}: {name}")
                            missing_entity = name
                    else:
                        missing_entity = name
                        logger.info(f"❌ Not found in database: {name}")
                
                except Exception as e:
                    logger.warning(f"Error querying {name}: {str(e)}")
                    missing_entity = name
                
                # Save results
                if found_entity:
                    result.found_entities.append(found_entity)
                if missing_entity:
                    result.missing_entities.append(missing_entity)
            
        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            raise Exception(f"Database query error: {str(e)}")
        
        return result
    
    def _run(
        self, 
        image_id: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None,
        runtime: Optional[ToolRuntime] = None
    ) -> Tuple[str, SearchingResult]:
        """
        Run the entity recognition tool.
        
        Args:
            image_id: UUID of the image file
            runtime: LangGraph ToolRuntime context
            
        Returns:
            String result message
        """
        try:
            # Step 1: Extract entities from image
            logger.info(f"✅ Resolving image for entity recognition: {image_id}")
            
            user_id = "default_user"
            thread_id = "default_thread"
            media_registry = None
            if runtime and runtime.config:
                configurable = runtime.config.get("configurable", {})
                user_id = str(configurable.get("user_id", "default_user"))
                thread_id = str(configurable.get("thread_id", "default_thread"))
                media_registry = configurable.get("media_registry")

            if not media_registry:
                raise ValueError("MediaRegistryService not found in runtime config")

            entities = self._extract_entities_from_image(image_id, media_registry=media_registry, user_id=user_id, thread_id=thread_id)
            
            if not entities:
                logger.info("No entities detected in image")
                return "This tool can't detect any human faces for recognition. So you should ask user to rephare the description about the image  ", SearchingResult()
            
            logger.info(f"Extracted {len(entities)} entities from image")
            
            # Step 2: Query database for detailed information
            db_result = self._query_database(entities)
            
            # Step 3: Format response
            found_names = ", ".join([entity.NAME for entity in db_result.found_entities]) if db_result.found_entities else ""
            missing_names = ", ".join(db_result.missing_entities) if db_result.missing_entities else ""
            
            parts = []
            if found_names:
                parts.append(f"Found detail information of {len(found_names)} entities: {found_names}.")
            if missing_names:
                parts.append(f"Not found information of {len(missing_names)} entities in the database: {missing_names}.")
             
            parts.append("The information for the found entities has been saved to temporary memory for use by other tools.")
            
            response_msg = "Successfully retrieved soccer-related entities. " + " ".join(parts)
            return response_msg, db_result
        except Exception as e:
            error_msg = f"Error in entity recognition: {str(e)}"
            logger.error(error_msg)

            # Send error to LangSmith run tree
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.end(
                    error=error_msg
                )

            return error_msg, SearchingResult()
    
    
    
