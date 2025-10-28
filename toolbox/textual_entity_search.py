import pymongo
import dns.resolver
import logging

from config import settings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from pymongo.server_api import ServerApi
from models.soccerwiki_entities import PlayerSchema, RefereeSchema, VenueSchema, TeamSchema

from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


# Setup logger
logger = logging.getLogger(__name__)

class SoccerEntities(BaseModel):
    """
    A schema for extracting soccer-related entities from a text query.
    """
    unknown: Optional[List[str]] = Field(default=None, description="List of entities that not sure about their type")
    player: Optional[List[str]] = Field(default=None, description="List of player names mentioned in the query")
    team: Optional[List[str]] = Field(default=None, description="List of team names mentioned in the query")
    venue: Optional[List[str]] = Field(default=None, description="List of venue names mentioned in the query")
    referee: Optional[List[str]] = Field(default=None, description="List of referee names mentioned in the query")

class SearchingResult(BaseModel):
    """
    Schema for searching entities information from database
    """
    found_entities: List[PlayerSchema | RefereeSchema | VenueSchema | TeamSchema] = Field(default_factory=list, description="Entities found in database\\other sources")
    missing_entities: List[str] = Field(default_factory=list, description="Entities not found in database\\other sources")


def extract_entity(query: str) -> Optional[SoccerEntities]:
    """
    Extract soccer-related entities from the user's query using LLM.

    Args:
        query (str): User's soccer-related query
    Returns:
        SoccerEntities: Pydantic Object or None if error occurs
    """

    # Define prompt template
    extract_entity_prompt = ChatPromptTemplate.from_messages([
      SystemMessagePromptTemplate.from_template(
          "You are a helpful assistant that extracts soccer-related entities from user queries. "
          "Extract only entities that are clearly mentioned in the query. "
          "Be precise and extract the exact names as they appear."
      ),
          HumanMessagePromptTemplate.from_template(
                """# TASK: Your task is to analyze the # INPUT and perform Named Entity Recognition (NER). You must:
    1. Identify the specific entity mentioned.
    2. Determine the correct ENTITY TYPE based on the context provided in the # INPUT.
    3. Extract the exact name of the entity.
    # CONTEXT AND AMBIGUITY RESOLUTION:
    When a proper noun is mentioned, its entity type can often be ambiguous (e.g., 'A' could be a 'player' or a 'referee'). Therefore, you must use the full context of the # INPUT to infer the correct entity type. Crucially, if the context in the # INPUT is insufficient to confidently classify the entity, or if you are uncertain, you must classify the entity as 'unknown'. Prioritize 'unknown' in cases of ambiguity or low confidence.**
    # ENTITY TYPE
    The defined entity types are: **player**, **referee**, **team**, **venue**, **unknown**.
    Note: If the entity is a **coach**, it must be classified as a **player**.
    # OUTPUT FORMAT
    The extracted entity name must exactly match its appearance in the question.
    The output must adhere strictly to the following JSON format:
    
    {output_format}
    # INPUT: {question}
    # EXAMPLES
    Example 1: Ambiguity leading to 'unknown'
    INPUT: How many goals did Messi score?
    # OUTPUT:
    {{
    "unknown": ["Messi"],
    "player": null,
    "team": null,
    "venue": null,
    "referee": null
    }}
    
    ### Example 2: Clear context and multiple entities 
    # INPUT: What is the salary of Messi player in Paris Saint-Germain?
    # OUTPUT:
    {{
    "unknown": null,
    "player": ["Messi"],
    "team": ["Paris Saint-Germain"],
    "venue": null,
    "referee": null
    }}
    
    ### Example 4:
    # INPUT: Which referee officiated the match at Wembley Stadium?
    # OUTPUT:
    {{
    "unknown": null,
    "player": null,
    "team": null,
    "venue": ["Wembley Stadium"],
    "referee": null
    }}
    
    ### Example 5: Highly ambiguous entity
    # INPUT: When did Zidane start his career?
    # OUTPUT:
    {{
    "unknown": ["Zidane"],
    "player": null,
    "team": null,
    "venue": null,
    "referee": null
    }}
    """),
    ])

    # Create output parser
    parser = PydanticOutputParser(pydantic_object=SoccerEntities)

    # Call LLM
    model = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest", 
        temperature=0.5,  
        top_p=0.95
    )
    
    # Combine prompt, model, and parser
    extract_entity_chain = extract_entity_prompt | model | parser

    try:
        soccer_entities = extract_entity_chain.invoke({
            "output_format": parser.get_format_instructions(),
            "question": query
        })

        # Log extracted entities
        logging.info(f"Extracted entities: {soccer_entities}")
        
        return soccer_entities
        
    except Exception as e:
        logging.error(f"Error extracting entities from query '{query}': {str(e)}")
        return None

def query_database(soccer_entities: SoccerEntities) -> SearchingResult:
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
        dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
        dns.resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']  
        client = pymongo.MongoClient(mongo_srv, server_api=ServerApi('1'))
        db = client.get_database(name=database_name)
        collection = db.get_collection(name=collection_name)
        
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
                            "FULL_NAME": entity_name,
                        }
                    else:
                        filter = { 
                            "$and": [
                                {
                                    "ENTITY_TYPE": entity_type,
                                },
                                {
                                    "FULL_NAME": entity_name,
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
                            parsed_entity = parse_entity_result(entity_data)
                            if parsed_entity:
                                found_entities.append(parsed_entity)
                                found = True
                                logging.info(f"Found and parsed {entity_type}: {entity_name}")
                            else:
                                logging.info(f"Failed to parse {entity_type}: {entity_name}")
                            
                    except Exception as e:
                        logging.warning(f"Error querying {entity_name} in {collection_name}: {str(e)}")

                    if not found:
                        missing_entities.append(entity_name)
                        logging.info(f"Not found {entity_type}: {entity_name}")

                # Save results
                if found_entities:
                    result.found_entities.extend(found_entities)
                if missing_entities:
                    result.missing_entities.extend(missing_entities)

        # Close connection
        client.close()
        
    except Exception as e:
        logging.error(f"Error querying database: {str(e)}")
    
    return result


def parse_entity_result(entity_data: dict) -> Optional[BaseModel]:
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
@tool
def textual_entity_search(query: str) -> SearchingResult:  
    """
    Given question about soccer-related entities (player, team, etc.), the tool retrieves the requiring entities of the question, and return its according WikiPage. The entity database contains the history and background knowledge for all the players, teams, venues, coaches and referees from games are from 2022 World Cup and 6 European major leagues (England Premier, Germany Bundesliga, Italy Serie-a, Spain Laliga, France Ligue-1 and European Champions League) during 2017-2024.
    
    Args:
        query (str): Prompt query could be the original question.
        
    Returns:
        str: Information about found entities or error message
    """
    try:
        # Extract entities from query
        entities = extract_entity(query)
        logger.info(f"Extracted entities: {entities}")
        if not entities:
            logger.info("No entities extracted from query.")
            return SearchingResult()
        
        # Query database for extracted entities
        db_searching_result = query_database(entities)
        logger.debug(f"Database searching result: {db_searching_result}")

        return db_searching_result

    except Exception as e:
        logging.error(f"Error in textual_entity_search: {str(e)}")
        return SearchingResult()
