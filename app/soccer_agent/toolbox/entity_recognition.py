import json
import os
import logging
import base64
import pymongo
import numpy as np
from collections import defaultdict
import requests

from dns import resolver
from langsmith import get_current_run_tree
from pymongo.server_api import ServerApi
from typing import Any, Tuple, Type, Optional, Literal, List, Dict
from pydantic import BaseModel, Field, PrivateAttr
from qdrant_client import QdrantClient, models
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from app.config.settings import settings
from app.config.config import QDRANT_SEARCH_SCORE_THRESHOLD as THRESHOLD
from app.schema.textual_entity_search import SearchingResult
from app.schema.soccerwiki_entities import PlayerSchema, RefereeSchema, VenueSchema, TeamSchema
from app.cache.exact_cache import exact_cache


# Setup logger
logger = logging.getLogger(__name__)

class EntityRecognitionInput(BaseModel):
    material: List[str] = Field(..., description="Paths to image files with entity_recognition names; if omitted, defaults to all image paths.")

class EntityRecognitionTool(BaseTool):
    """
    Tool for recognizing soccer entities from images using face recognition.
    """
    
    name: str = "entity_recognition"
    description: str = """Given an image path, the tool retrieves the requiring entities of the question, and return its according WikiPage. You should use segment tool before this tool. The entity database contains the history and background knowledge for all the players, teams, venues, coaches and referees from games are from 2022 World Cup and 6 European major leagues (England Premier, Germany Bundesliga, Italy Serie-a, Spain Laliga, France Ligue-1 and European Champions League) during 2017-2024."""
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    args_schema: Type[BaseModel] = EntityRecognitionInput # type: ignore
    
    # Internal state (not exposed to LLM)
    _qdrant_client: Optional[QdrantClient] = PrivateAttr(default=None)
    _mongo_client: Optional[pymongo.MongoClient] = PrivateAttr(default=None)
    _deepface_rep: Any = PrivateAttr(default=None)

    # Azure DeepFace Endpoint attributes
    _df_endpoint_uri: str = PrivateAttr(default="")
    _df_payload_header: Dict[str, str] = PrivateAttr(default={})
    _df_endpoint_key: str = PrivateAttr(default="")

    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
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
        self._df_endpoint_uri = settings.DEEPFACE_ENDPOINT_URI
        self._df_endpoint_key = settings.DEEPFACE_ENDPOINT_KEY
        
        # Validate endpoint configuration
        if not self._df_endpoint_uri:
            logger.error(f"DEEPFACE_ENDPOINT_URI is empty or not configured. Value: '{self._df_endpoint_uri}'")
            raise ValueError("DEEPFACE_ENDPOINT_URI is not configured. Please check settings.")
        if not self._df_endpoint_key:
            logger.error(f"DEEPFACE_ENDPOINT_KEY is empty or not configured.")
            raise ValueError("DEEPFACE_ENDPOINT_KEY is not configured. Please check settings.")
        
        logger.info(f"DeepFace Endpoint URI: {self._df_endpoint_uri}")
        
        self._df_payload_header = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._df_endpoint_key}'
        }
        response = requests.post(url=self._df_endpoint_uri, headers=self._df_payload_header, timeout=60)
        if response.status_code == 200:
            logger.info("✅ DeepFace endpoint is reachable")
        else:
            raise ConnectionError(f"Failed to connect to DeepFace endpoint: {response.status_code} - {response.text}")

    
    def _create_payload(self, image_path: str, max_faces: int = 5, confidence_threshold: float = 0.85) -> Dict:
        """
        Create payload for DeepFace endpoint.
        
        :param image_path: Path to the image file
        :type image_path: str
        :param max_faces: Maximum number of faces to detect
        :type max_faces: int
        :param confidence_threshold: Confidence threshold for face detection
        :type confidence_threshold: float
        :return: JSON payload for the request
        :rtype: Dict
        """

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        

        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "image": image_base64,
            "max_faces": max_faces,
            "confidence_threshold": confidence_threshold
        }
  
        return json.dumps(payload)
    
    @staticmethod    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _extract_entities_from_image(self, image_path: str) -> List[Dict]:
        """
        Extract soccer entities from image using face recognition.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of entity dictionaries with ENTITY_TYPE and NAME
        """
        collection_name = settings.QDRANT_COLLECTION_NAME 
        assert collection_name is not None, "Qdrant client is not initialized"
        
        # Create payload for DeepFace endpoint
        payload = self._create_payload(image_path=image_path, max_faces=5, confidence_threshold=0.85)

        try:
        # Call to DeepFace Endpoint
            response: requests.Response = requests.post(
                url=self._df_endpoint_uri,
                headers=self._df_payload_header,
                data=payload,
                timeout=60
            )
        except requests.RequestException as e:
            raise Exception(f"DeepFace endpoint request error: {str(e)}")
        response_dict: Dict = response.json()


        # Handle response
        if response.status_code != 200:
            raise Exception(f"DeepFace endpoint request failed: {response.status_code} - {response.text}")
        if response_dict.get("success", False) is False:
            raise Exception(f"DeepFace endpoint error: {response_dict.get('error', 'Unknown error')} ")

        results = response_dict.get("result") or []
        logger.info(f"✅ DeepFace endpoint response received successfully. There are {len(results)} results")
        # Print out each result for debugging
        for idx, res in enumerate(results):
            logger.info(f"Result {idx+1}: {res.get('facial_area', {})}, Confidence: {res.get('face_confidence', 0.0)}")
        
        # Search entities in Qdrant
        detected_faces = results
        soccer_entities = []
        for idx, detected_face in enumerate(detected_faces):
            detected_face_embedding = detected_face.get("embedding", [])
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
                return None, "Can't find any matching entities in the database."
            
            if ranked_candidates[0]:
                soccer_entities.append({
                    "ENTITY_TYPE": candidates[ranked_candidates[0][0]]["ENTITY_TYPE"],
                    "NAME": ranked_candidates[0][0]
                })
                logger.info(f"Face {idx+1}: Matched to {ranked_candidates[0]}")
        
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
    
    @exact_cache.cache(ttl=60 * 60, validatedModel=SearchingResult)
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
    
    def _run(self, material: List[str], run_manager: Optional[CallbackManagerForToolRun] = None) -> Tuple[str, SearchingResult]:
        """
        Run the entity recognition tool.
        
        Args:
            material: Paths to the image files
            
        Returns:
            String result message
        """
        try:
            # Step 1: Extract entities from image
            for material_path in material:
                if not os.path.isfile(material_path):
                    raise FileNotFoundError(f"Material file not found: {material_path}")
            entities = self._extract_entities_from_image(material[0]) #TODO: fix to support multiple images
            
            if not entities:
                logger.info("No entities detected in image")
                return "No soccer-related entities found in the image.", SearchingResult()
            
            logger.info(f"Extracted {len(entities)} entities from image")
            
            # Step 2: Query database for detailed information
            db_result = self._query_database(entities)
            
            # Step 3: Format response
            found_names = ", ".join([entity.NAME for entity in db_result.found_entities]) if db_result.found_entities else ""
            missing_names = ", ".join(db_result.missing_entities) if db_result.missing_entities else ""
            
            parts = []
            if found_names:
                parts.append(f"Found entities: {found_names}.")
            if missing_names:
                parts.append(f"Missing entities: {missing_names}.")
             
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

            return f"Error occurred while processing image: {str(e)}", SearchingResult()
    
    
    
