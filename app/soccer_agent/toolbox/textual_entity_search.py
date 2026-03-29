import pymongo
from dns import resolver
import logging
from app.config import settings
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Type, Literal
from pymongo.server_api import ServerApi

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langsmith import get_current_run_tree

from app.schema.soccerwiki_entities import PlayerSchema, RefereeSchema, VenueSchema, TeamSchema
from app.schema.textual_entity_search import SoccerEntities, SearchingResult
from app.cache.standard_cache import standard_cache

# Setup logger
logger = logging.getLogger(__name__)

class TextualEntitySearchInput(BaseModel):
    entity_names: List[str] = Field(
        ...,
        description=(
            "List of soccer-related entity names already extracted from the user's question or inferred by previous tools results. "
            "Each item should be a direct name (player, team, venue, referee, coach, club) without extra narration. "
            "Use this tool only after the agent has resolved the names; do not pass raw user questions here."
            "Remember to capitalize the first letter of each word in the entity names."
        ),
        examples=[
            ["Lionel Messi", "Barcelona"],
            ["Kylian Mbappe", "Parc des Princes"],
            ["Old Trafford"],
        ],
    )


class TextualEntitySearchTool(BaseTool):
    name: str = "textual_entity_search"
    description: str = """
    Given question about soccer-related entities (player, team, etc.), the tool retrieves the requiring entities of the question, and return its according WikiPage. The entity database contains the history and background knowledge for all the players, teams, venues, coaches and referees from games are from 2022 World Cup and 6 European major leagues (England Premier, Germany Bundesliga, Italy Serie-a, Spain Laliga, France Ligue-1 and European Champions League) during 2017-2024.
    """
    args_schema: Type[BaseModel] = TextualEntitySearchInput # type: ignore
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    


    def __init__(self):
        super().__init__()

    def _run(self, entity_names: List[str], run_manager: Optional[CallbackManagerForToolRun] = None) -> Tuple[str, SearchingResult]:
        run_tree = get_current_run_tree()
        try:
            if not entity_names:
                logger.info("No entity names provided to textual_entity_search.")
                return "This tool can't find any soccer-related entities in the provided input.", SearchingResult()

            # Execution Agent supplies resolved names; wrap them as unknown type for DB lookup
            entities = SoccerEntities(unknown=entity_names)
            logger.info(f"Received entities for lookup: {entities}")
            
            # Query database for provided entities
            db_searching_result = TextualEntitySearchTool._query_database(entities)
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
            error_msg = f"Error in textual_entity_search: {str(e)}"
            logging.error(error_msg, exc_info=True)

            # Send error to LangSmith run tree
            if run_tree:
                run_tree.end(
                    error=error_msg
                )
            
            # Return detailed error message to the Agent
            return f"An error occurred while executing the tool. Details: {str(e)}. Please retry the tool or stop the process.", SearchingResult()

        
    @staticmethod
    def _parse_entity_result(entity_data: Dict) -> Optional[BaseModel]:
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
                error_msg = f"Failed to parse {entity_type}: {str(e)}"
                logger.warning(error_msg)
                raise e
        
        logger.warning(f"Unknown entity type: {entity_type}")
        return None
      
    @staticmethod
    @standard_cache.cache(ttl=60 * 60, validatedModel=SearchingResult)
    def _query_database(soccer_entities: SoccerEntities) -> SearchingResult:
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
                raise ValueError("MONGO_SRV configuration not found. Please set it in the application settings.")

            # Try to use preloaded MongoDB client from main.py
            client = None
            try:
                import main
                if main.mongo_client is not None:
                    client = main.mongo_client
                    logger.info("✅ Using preloaded MongoDB client from lifespan")
            except (ImportError, AttributeError):
                pass
            
            # Fallback: Create new connection if not preloaded
            if client is None:
                logger.info("Creating MongoDB connection on demand...")
                resolver.default_resolver = resolver.Resolver(configure=False)
                resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']  
                client = pymongo.MongoClient(mongo_srv, server_api=ServerApi('1'))
            
            db = client.get_database(name=database_name)
            collection = db.get_collection(name=collection_name) # TODO: Check the connection status
            
            # Iterate through each entity type
            for entity_type in SoccerEntities.model_fields.keys():
                logger.info(f"Searching entity type = {entity_type}")
                entities_list = getattr(soccer_entities, entity_type, None)
                
                if entities_list:
                    found_entities = []
                    missing_entities = []

                    # Iterate through each entity in the database
                    for entity_name in entities_list:
                        logger.info(f"Searching {entity_type}: {entity_name}")

                        # Search in database
                        if entity_type == "unknown":
                            filter = { 
                                "NAME": {"$regex": entity_name, "$options": "i"},
                            }
                        else:
                            filter = { 
                                "$and": [
                                    {
                                        "ENTITY_TYPE": entity_type,
                                    },
                                    {
                                        "NAME": {"$regex": entity_name, "$options": "i"},
                                    }
                                ]# type: ignore
                            }

                        
                        found = False
                        try:
                            entity_data = collection.find_one(filter)
                            if entity_data:
                                logger.debug(f"Query result for {entity_name}: {entity_data}")

                                # Convert _id to string
                                if '_id' in entity_data:
                                    entity_data['_id'] = str(entity_data['_id'])
                                
                                # Parse to Pydantic model
                                parsed_entity = TextualEntitySearchTool._parse_entity_result(entity_data)
                                if parsed_entity:
                                    found_entities.append(parsed_entity)
                                    found = True
                                    logging.info(f"Found and parsed {entity_type}: {entity_name}")
                                else:
                                    logging.info(f"Failed to parse {entity_type}: {entity_name}")
                                
                        except Exception as e:
                            logging.warning(f"Error querying {entity_name} in {collection_name}: {str(e)}")
                            raise e

                        if not found:
                            missing_entities.append(entity_name)
                            logging.info(f"Not found {entity_type}: {entity_name}")

                    # Save results
                    if found_entities:
                        result.found_entities.extend(found_entities)
                    if missing_entities:
                        result.missing_entities.extend(missing_entities)

            # Only close connection if we created it (not using preloaded one)
            try:
                import main
                if client is not main.mongo_client:
                    client.close()
                    logger.info("Local MongoDB connection closed")
            except (ImportError, AttributeError):
                client.close()
            
            return result
        
        except Exception as e:
            error_msg = f"Error querying database: {str(e)}"
            logging.error(error_msg)
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.end(
                    error=error_msg
                )
            # Raise with context
            raise RuntimeError(error_msg) from e
        

