import os
import logging
from typing import List, Optional, Dict, Tuple, Annotated

from dns import resolver
import pymongo
from pymongo.server_api import ServerApi

from deepface import DeepFace
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain.tools import tool

from app.config import settings
from app.schema.soccerwiki_entities import (
    PlayerSchema,
    RefereeSchema,
    VenueSchema,
    TeamSchema,
)
from app.schema.toolbox.textual_entity_search import (
    SoccerEntities,
    SearchingResult,
)
from app.toolbox.textual_entity_search import parse_entity_result

# Environment variables
os.environ["TF_USE_LEGACY_KERAS"] = "1"        # Use legacy Keras
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # Disable oneDNN rounding warnings

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)


_qdrant_client: Optional[QdrantClient] = None

def get_qdrant_client() -> QdrantClient: # Move to settings
    """Get or create singleton Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be set in .env file"
            )
        
        _qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        logger.info("✅ Qdrant client initialized successfully")
    
    return _qdrant_client

def extract_entity(image_path: str) -> list:
    """
    Extract soccer-related entities from the user's query using LLM.

    Args:
        image_path (str): Path to the input file
    Returns:
        list: List of search results for each detected face
    """
    
    client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "SoccerAgent")
    embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend="retinaface",
            normalization="Facenet",
            enforce_detection=False,
            max_faces = 15
        )
        
    # Filter faces with confidence > 0.85
    valid_faces = []
    for face_data in embedding_objs:
        if face_data.get('face_confidence', 0) > 0.85:
            valid_faces.append(face_data['embedding'])
    
    # Search for all valid faces
    soccer_entities = []
    for embedding in valid_faces:
        search_result = client.query_points(
                collection_name=collection_name,
                query=embedding,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(rescore=True),
                    exact=True
                ),
                limit=1,
                score_threshold=0.65
            )
        if search_result.points:
            soccer_entities.append(search_result.points[0].payload)
    return soccer_entities

def query_database(soccer_entities: list) -> SearchingResult:
    """
    Query MongoDB database for information on the extracted soccer entities.
    Args:
        soccer_entities: Extracted entities using extract_entity()

    Returns:
        SearchingResult: Query results including found and missing entities
    """

    # Initialize result which will hold found and missing entities
    result = SearchingResult()
    try:
        # Get MongoDB connection info from config
        mongo_srv = settings.MONGO_SRV
        database_name = settings.SOCCER_DB_NAME
        collection_name = settings.SOCCER_COLLECTION_NAME
        
        if not mongo_srv:
            logging.error("MONGO_SRV configuration not found")
            return result

        # Connect to MongoDB
        resolver.default_resolver = resolver.Resolver(configure=False)
        resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']  
        client = pymongo.MongoClient(mongo_srv, server_api=ServerApi('1'))
        db = client.get_database(name=database_name)
        collection = db.get_collection(name=collection_name)
        
        # Iterate through each entity type
        for entity in soccer_entities:
            found_entity = None
            missing_entity = None
            logger.info(f"Searching entity type = {entity['ENTITY_TYPE']}")
            entity_type = entity['ENTITY_TYPE']
            name = entity['NAME']
            filter = { 
                "$and": [
                    {
                        "ENTITY_TYPE": entity_type,
                    },
                    {
                        "NAME": name,
                    }
                ]# type: ignore
            }

                    
            found = False
            try:
                entity_data = collection.find_one(filter)
                if entity_data:
                    logger.debug(f"Query result for {name}: {entity_data}")

                    # Convert _id to string
                    if '_id' in entity_data:
                        entity_data['_id'] = str(entity_data['_id'])
                    
                    # Parse to Pydantic model
                    parsed_entity = parse_entity_result(entity_data)
                    if parsed_entity:
                        found_entity = parsed_entity
                        found = True
                        logging.info(f"Found and parsed {entity_type}: {name}")
                    else:
                        logging.info(f"Failed to parse {entity_type}: {name}")

            except Exception as e:
                logging.warning(f"Error querying {name} in {collection_name}: {str(e)}")

            if not found:
                missing_entity = name
                logging.info(f"Not found {entity_type}: {name}")

            # Save results (moved outside the if not found block)
            if found_entity:
                result.found_entities.append(found_entity)
            if missing_entity:
                result.missing_entities.append(missing_entity)

        # Close connection
        client.close()
        
    except Exception as e:
        logging.error(f"Error querying database: {str(e)}")
    
    return result

@tool(
        response_format='content_and_artifact',
)
def entity_recognition(image_path: str) -> Tuple[str, SearchingResult]:  
    """
    Given an image path, the tool retrieves the soccer-related entities present in the image and returns their corresponding WikiPage. The entity database contains the history and background knowledge for all the players, teams, venues, coaches, and referees from games are from 2022 World Cup and 6 European major leagues (England Premier, Germany Bundesliga, Italy Serie-a, Spain Laliga, France Ligue-1 and European Champions League) during 2017-2024.

    Args:
        image_path (str): The path to the image file.
    """
    try:
        # Extract entities from query
        entities = extract_entity(image_path)
        logger.info(f"Extracted entities: {entities}")
        if not entities:
            logger.info("No entities extracted from query.")
            return "This tool can't find any soccer-related entities in the user query.", SearchingResult()
        
        # Query database for extracted entities
        db_searching_result = query_database(entities)
        logger.debug(f"Database searching result: {db_searching_result}")

        # Prepare response message for Execution Agent
        found_names = ", ".join([entity.NAME for entity in db_searching_result.found_entities]) if db_searching_result.found_entities else ""
        missing_names = ", ".join(db_searching_result.missing_entities) if db_searching_result.missing_entities else ""

        parts = []
        if found_names:
            parts.append(f"Found entities: {found_names}.")
        if missing_names:
            parts.append(f"Missing entities: {missing_names}.")

        parts.append("The information for the found entities has been saved to temporary memory for use by other tools. You may proceed to run the next tool as planned.")

        response_msg = "Successfully retrieved soccer-related entities. " + " ".join(parts)
        return response_msg, db_searching_result

    except Exception as e:
        logging.error(f"Error in entity_recognition: {str(e)}")
        return "Error occurred while searching for entities.", SearchingResult()


if __name__ == "__main__":
    """Test the entity recognition pipeline."""
    import sys
    from pathlib import Path
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*70)
    print("🧪 TESTING ENTITY RECOGNITION")
    print("="*70)
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        test_image_path = "D:\\Project\\FinalProject\\example_images\\abdukodir-khusanov-man-city.jpg"
        print(f"💡 No image path provided. Using default: {test_image_path}")
        print(f"   Usage: python entity_recognition.py <path_to_image>")
        print()
    
    # Check if file exists
    if not Path(test_image_path).exists():
        print(f"❌ Error: File '{test_image_path}' not found!")
        sys.exit(1)
    
    print(f"📁 Image path: {test_image_path}")
    print()
    
    try:
        # Step 1: Extract entities from image
        print("-"*70)
        print("STEP 1: Extracting entities from image")
        print("-"*70)
        
        entities = extract_entity(test_image_path)
        
        print(f"✅ Detected {len(entities)} entities with confidence > 0.85")
        for idx, entity in enumerate(entities, 1):
            print(f"\n👤 Entity #{idx}:")
            print(f"   Type: {entity.get('ENTITY_TYPE', 'Unknown')}")
            print(f"   Name: {entity.get('NAME', 'Unknown')}")
        print()
        
        if not entities:
            print("⚠️  No entities detected. Test stopped.")
            sys.exit(0)
        
        # Step 2: Query database for entities
        print("-"*70)
        print("STEP 2: Querying database for entity information")
        print("-"*70)
        
        db_result = query_database(entities)
        
        print(f"✅ Found entities: {len(db_result.found_entities)}")
        print(f"⚠️  Missing entities: {len(db_result.missing_entities)}")
        print(db_result)
        
        # Display found entities
        if db_result.found_entities:
            print("📊 Found Entities Details:")
            for entity in db_result.found_entities:
                print(f"\n   🏆 {entity.NAME}")
                print(f"      Type: {entity.ENTITY_TYPE}")
                if hasattr(entity, 'COUNTRY'):
                    print(f"      Country: {entity.COUNTRY}")
                if hasattr(entity, 'POSITION'):
                    print(f"      Position: {entity.POSITION}")
        
        # Display missing entities
        if db_result.missing_entities:
            print("\n⚠️  Missing Entities:")
            for name in db_result.missing_entities:
                print(f"   - {name}")
        
        print()
        print("-"*70)
        print("STEP 3: Testing full entity_recognition tool")
        print("-"*70)
        
        result = entity_recognition.invoke({"image_path": test_image_path})
        print()
        print("="*70)
        print("🎉 TEST COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        print()
        print("="*70)
        print("❌ ERROR OCCURRED")
        print("="*70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
