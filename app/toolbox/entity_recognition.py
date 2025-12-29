import os
import logging
from langsmith import get_current_run_tree
import pymongo
import numpy as np
from collections import defaultdict


from dns import resolver
from pymongo.server_api import ServerApi
from typing import Any, Tuple, Type, Optional, Literal,List, Dict
from pydantic import BaseModel, Field, PrivateAttr
from app.toolbox.helper.deepface import DeepFaceRepresentation
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun


from app.config import settings
from app.schema.textual_entity_search import SearchingResult
from app.schema.soccerwiki_entities import PlayerSchema, RefereeSchema, VenueSchema, TeamSchema

# Environment variables
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

class EntityRecognitionInput(BaseModel):
    material: List[str] = Field(..., description="Paths to image files with entity_recognition names; if omitted, defaults to all image paths.")

class EntityRecognitionTool(BaseTool):
    """
    Tool for recognizing soccer entities from images using face recognition.
    """
    
    name: str = "entity_recognition"
    description: str = """Given an image path, the tool retrieves the requiring entities of the question, and return its according WikiPage. The entity database contains the history and background knowledge for all the players, teams, venues, coaches and referees from games are from 2022 World Cup and 6 European major leagues (England Premier, Germany Bundesliga, Italy Serie-a, Spain Laliga, France Ligue-1 and European Champions League) during 2017-2024."""
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    args_schema: Type[BaseModel] = EntityRecognitionInput # type: ignore
    
    # Internal state (not exposed to LLM)
    _qdrant_client: Optional[QdrantClient] = None
    _mongo_client: Optional[pymongo.MongoClient] = None
    _deepface_rep: Any = PrivateAttr(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_clients()
        self.load_models()
    
    def _initialize_clients(self):
        """Initialize Qdrant and MongoDB clients as singletons."""
        # Initialize Qdrant client
        if self._qdrant_client is None:
            qdrant_url = settings.QDRANT_URL
            qdrant_api_key = settings.QDRANT_API_KEY
            
            self._qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
            logger.info("✅ Qdrant client initialized successfully")
        
        # Initialize MongoDB client
        if self._mongo_client is None:
            mongo_srv = settings.MONGO_SRV
            if mongo_srv:
                resolver.default_resolver = resolver.Resolver(configure=False)
                resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']
                self._mongo_client = pymongo.MongoClient(mongo_srv, server_api=ServerApi('1'))
                logger.info("✅ MongoDB client initialized successfully")
    def load_models(self):
        if self._deepface_rep is None:
            self._deepface_rep = DeepFaceRepresentation(
                model_recognition_name="Facenet512", 
                model_detector_name="retinaface"
            )
            logger.info("✅ DeepFace models loaded successfully")
    @staticmethod    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _extract_entities_from_image(self, image_path: str, THRESHOLD: int = 0.5) -> List[Dict]:
        """
        Extract soccer entities from image using face recognition.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of entity dictionaries with ENTITY_TYPE and NAME
        """
        collection_name = settings.QDRANT_COLLECTION_NAME
        
        # Extract face embeddings
        embedding_objs = self._deepface_rep.represent(
            img_path=image_path,
            model_name="Facenet512",
            detector_backend="retinaface",
            normalization="Facenet2018",
            enforce_detection=False,
            max_faces=15
        )
        
        # Filter faces with confidence > 0.85
        valid_faces = []
        for face_data in embedding_objs:
            if face_data.get('face_confidence', 0) > 0.85:
                valid_faces.append(face_data['embedding'])
        
        logger.info(f"Found {len(valid_faces)} valid faces with confidence > 0.85")
        
        # Search for all valid faces in Qdrant
        soccer_entities = []
        for idx, query_vector in enumerate(valid_faces):
            search_result = self._qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
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
                    score = self.cosine_similarity(query_vector, sub_vector)
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
                return None, "No match found"
            
            if ranked_candidates[0]:
                soccer_entities.append({
                    "ENTITY_TYPE": candidates[ranked_candidates[0][0]]["ENTITY_TYPE"],
                    "NAME": ranked_candidates[0][0]
                })
                logger.info(f"Face {idx+1}: Matched to {ranked_candidates[0]}")
            else:
                logger.info(f"Face {idx+1}: No match found")
        
        return soccer_entities
    def _parse_entity_result(self, entity_data: dict) -> Optional[BaseModel]:
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
        
        schema_map = {
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
                found_entity = None
                missing_entity = None
                
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
    
    
    def __del__(self):
        """Cleanup connections when object is destroyed."""
        try:
            if hasattr(self, '_mongo_client') and self._mongo_client is not None:
                self._mongo_client.close()
                logger.info("MongoDB client closed")
            if hasattr(self, '_qdrant_client') and self._qdrant_client is not None:
                self._qdrant_client.close()
                logger.info("Qdrant client cleanup completed")
        except Exception:
            # Silently ignore cleanup errors during shutdown
            pass
