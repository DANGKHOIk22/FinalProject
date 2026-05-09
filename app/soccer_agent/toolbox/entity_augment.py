import logging
from typing import Any, List, Literal, Optional, Tuple, Type

import pymongo
from dns import resolver
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langsmith import get_current_run_tree
from pydantic import BaseModel, Field, PrivateAttr
from pymongo.server_api import ServerApi

from app.cache.standard_cache import standard_cache
from app.config import settings
from app.schema.soccerwiki_entities import (
    PlayerSchema,
    RefereeSchema,
    TeamSchema,
    VenueSchema,
)
from app.schema.textual_entity_search import SearchingResult, SoccerEntities
from app.soccer_agent.factory.llm_provider import get_llm
from app.soccer_agent.prompts.toolbox.textual_retrieval_augment import (
    get_textual_retrieval_augment_prompt_template,
)

logger = logging.getLogger(__name__)


class EntityAugmentInput(BaseModel):
    entity_names: List[str] = Field(
        ...,
        description=(
            "List of soccer-related entity names extracted from the user's question or inferred by previous tools. "
            "Each item should be a direct name (player, team, venue, referee, coach, club) without extra narration. "
            "Capitalize the first letter of each word."
        ),
        examples=[
            ["Lionel Messi", "Barcelona"],
            ["Kylian Mbappe", "Parc des Princes"],
            ["Old Trafford"],
        ],
    )
    query: str = Field(
        description=(
            "The user's question (or a well-defined retrieval query) used to focus the final answer."
        ),
    )


class EntityAugmentTool(BaseTool):
    name: str = "entity_augment"
    description: str = """
    Look up soccer-related entities (players, teams, venues, coaches, referees) in the local
    knowledge base by name and synthesize a concise, user-facing answer in a single step.
    The dataset covers matches from 2017-2024, including the 2022 World Cup and six major
    European leagues (Premier League, Bundesliga, Serie A, LaLiga, Ligue 1, Champions League).
    Use this tool whenever the user asks about background, biography, history, achievements,
    or other entity-level facts. Returns the final synthesized answer as text; entity records
    found and any missing entity names are also returned as a structured artifact for tracing.
    """
    args_schema: Type[BaseModel] = EntityAugmentInput  # type: ignore
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    _llm: Any = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._llm = get_llm("retrieval-augment")

    def _run(
        self,
        entity_names: List[str],
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, SearchingResult]:
        run_tree = get_current_run_tree()
        try:
            if not entity_names:
                logger.info("No entity names provided to entity_augment.")
                return (
                    "No entity names were provided. Please supply at least one entity name.",
                    SearchingResult(),
                )

            entities = SoccerEntities(unknown=entity_names)
            logger.info(f"Received entities for lookup: {entities}")

            searching_result = EntityAugmentTool._query_database(entities)
            logger.info(
                f"DB lookup: found={len(searching_result.found_entities)}, "
                f"missing={len(searching_result.missing_entities)}"
            )

            answer_text = self._generate_answer(query, searching_result)
            logger.info(f"✅ entity_augment: answer={answer_text[:200]}")
            return answer_text, searching_result

        except Exception as e:
            error_msg = f"Error in entity_augment: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if run_tree:
                run_tree.end(error=error_msg)
            return (
                f"An error occurred while executing entity_augment. Details: {str(e)}. "
                "Please retry with modified inputs or stop the process.",
                SearchingResult(),
            )

    def _generate_answer(self, query: str, searching_result: SearchingResult) -> str:
        """Run the retrieval-augment LLM chain on the DB lookup result."""
        run_tree = get_current_run_tree()
        prompt_template = get_textual_retrieval_augment_prompt_template()
        chain = prompt_template | self._llm

        searching_result_text = self._aggregate_searching_results(searching_result)
        inputs = {"query": query, "searching_result": searching_result_text}

        try:
            response = chain.invoke(inputs)
            return str(response.text)
        except Exception as e:
            error_msg = f"Error generating answer in entity_augment: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if run_tree:
                run_tree.end(error=error_msg)
            return (
                "An error occurred while generating the answer based on the retrieved information."
            )

    @staticmethod
    def _aggregate_searching_results(searching_result: SearchingResult) -> str:
        """Aggregate searching results into a textual format for the LLM prompt."""
        aggregated_text = ""
        for entity in searching_result.found_entities:
            aggregated_text += "-" * 10 + "\n"
            aggregated_text += f"INFORMATION ABOUT:  {entity.NAME}:\n"
            aggregated_text += f"(ENTITY TYPE: {entity.ENTITY_TYPE})\n"
            if getattr(entity, "SUMMARY", None):
                aggregated_text += f"SUMMARY: {entity.SUMMARY}\n"
            if getattr(entity, "INFOBOX", None):
                aggregated_text += f"INFOBOX: {entity.INFOBOX}\n"
            if getattr(entity, "CONTENT", None):
                aggregated_text += f"CONTENT: {entity.CONTENT}\n"
            aggregated_text += "-" * 10 + "\n\n"

        if searching_result.missing_entities:
            aggregated_text += "NOT FOUND INFORMATION FOR THE FOLLOWING ENTITIES: "
            aggregated_text += ", ".join(searching_result.missing_entities) + "\n"

        return aggregated_text

    @staticmethod
    def _parse_entity_result(entity_data: dict) -> Optional[BaseModel]:
        """Parse a MongoDB document into the appropriate Pydantic schema."""
        entity_type = entity_data.get("ENTITY_TYPE")
        if not entity_type:
            logger.warning("No ENTITY_TYPE found in entity document")
            return None

        schema_map = {
            "venue": VenueSchema,
            "player": PlayerSchema,
            "team": TeamSchema,
            "referee": RefereeSchema,
        }
        schema_class = schema_map.get(entity_type)
        if not schema_class:
            logger.warning(f"Unknown entity type: {entity_type}")
            return None

        try:
            return schema_class(**entity_data)
        except Exception as e:
            logger.warning(f"Failed to parse {entity_type}: {str(e)}")
            raise

    @staticmethod
    @standard_cache.cache(ttl=60 * 60, validatedModel=SearchingResult)
    def _query_database(soccer_entities: SoccerEntities) -> SearchingResult:
        """Query MongoDB for the given entities."""
        result = SearchingResult()
        try:
            mongo_srv = settings.MONGO_SRV
            database_name = settings.SOCCER_DB_NAME
            collection_name = settings.SOCCER_COLLECTION_NAME

            if not mongo_srv:
                raise ValueError(
                    "MONGO_SRV configuration not found. Please set it in the application settings."
                )

            client = None
            try:
                import main

                if main.mongo_client is not None:
                    client = main.mongo_client
                    logger.info("✅ Using preloaded MongoDB client from lifespan")
            except (ImportError, AttributeError):
                pass

            if client is None:
                logger.info("Creating MongoDB connection on demand...")
                resolver.default_resolver = resolver.Resolver(configure=False)
                resolver.default_resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
                client = pymongo.MongoClient(mongo_srv, server_api=ServerApi("1"))

            db = client.get_database(name=database_name)
            collection = db.get_collection(name=collection_name)

            for entity_type in SoccerEntities.model_fields.keys():
                entities_list = getattr(soccer_entities, entity_type, None)
                if not entities_list:
                    continue

                for entity_name in entities_list:
                    if entity_type == "unknown":
                        filter_q = {"NAME": {"$regex": entity_name, "$options": "i"}}
                    else:
                        filter_q = {
                            "$and": [
                                {"ENTITY_TYPE": entity_type},
                                {"NAME": {"$regex": entity_name, "$options": "i"}},
                            ]
                        }

                    found = False
                    try:
                        entity_data = collection.find_one(filter_q)
                        if entity_data:
                            if "_id" in entity_data:
                                entity_data["_id"] = str(entity_data["_id"])
                            parsed_entity = EntityAugmentTool._parse_entity_result(entity_data)
                            if parsed_entity:
                                result.found_entities.append(parsed_entity)
                                found = True
                                logger.info(f"Found and parsed {entity_type}: {entity_name}")
                            else:
                                logger.info(f"Failed to parse {entity_type}: {entity_name}")
                    except Exception as e:
                        logger.warning(
                            f"Error querying {entity_name} in {collection_name}: {str(e)}"
                        )
                        raise

                    if not found:
                        result.missing_entities.append(entity_name)
                        logger.info(f"Not found {entity_type}: {entity_name}")

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
            logger.error(error_msg)
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.end(error=error_msg)
            raise RuntimeError(error_msg) from e
