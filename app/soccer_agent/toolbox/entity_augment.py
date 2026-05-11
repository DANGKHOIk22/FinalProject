import asyncio
import logging
import re
from typing import Any, List, Literal, Optional, Tuple, Type

import pymongo
from dns import resolver
from langchain.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
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
from app.soccer_agent.services.content_cleaner import clean_wiki_markdown, extract_summary
from app.soccer_agent.services.tavily_service import TavilyService
from app.soccer_agent.toolbox._config_loader import tool_description

logger = logging.getLogger(__name__)

_ENTITY_URL_FIELD = {
    "player": "PLAYER_URL",
    "team": "TEAM_URL",
    "venue": "VENUE_URL",
    "referee": "REFEREE_URL",
}

# Trailing/leading organizational suffixes common in club names
_TEAM_SUFFIX_RE = re.compile(
    r"\s+(?:F\.?C\.?|A\.?F\.?C\.?|C\.?F\.?|S\.?C\.?|B\.?C\.?|E\.?C\.?|A\.?C\.?|S\.?S\.?C\.?|F\.?K\.?|B\.?K\.?)$",
    re.IGNORECASE,
)
_TEAM_PREFIX_RE = re.compile(r"^(?:F\.?C\.?|A\.?F\.?C\.?)\s+", re.IGNORECASE)


def _name_regex_variants(name: str) -> list[str]:
    """Return regex patterns to try for a given entity name.

    Includes the original name plus versions with common team suffixes/prefixes
    stripped so 'Manchester City FC' also matches 'Manchester City' in the DB.
    """
    variants: list[str] = [re.escape(name)]
    stripped = _TEAM_SUFFIX_RE.sub("", name).strip()
    if stripped and stripped != name:
        variants.append(re.escape(stripped))
    stripped2 = _TEAM_PREFIX_RE.sub("", name).strip()
    if stripped2 and stripped2 != name and stripped2 not in (name, stripped):
        variants.append(re.escape(stripped2))
    return variants

# Vietnamese type-label → SoccerEntities field name
# Order matters: longer/more-specific labels first to avoid partial matches.
_QUERY_TYPE_LABELS: list[tuple[str, str]] = [
    # Vietnamese (with diacritics)
    ("huấn luyện viên", "player"),
    ("câu lạc bộ", "team"),
    ("đội tuyển quốc gia", "team"),
    ("đội tuyển", "team"),
    ("đội bóng", "team"),
    ("cầu thủ", "player"),
    ("thủ môn", "player"),
    ("tiền đạo", "player"),
    ("hậu vệ", "player"),
    ("tiền vệ", "player"),
    ("trọng tài", "referee"),
    ("sân vận động", "venue"),
    ("hlv", "player"),
    ("clb", "team"),
    ("đội", "team"),
    ("sân", "venue"),
    # Vietnamese (no diacritics / ASCII)
    ("huan luyen vien", "player"),
    ("cau lac bo", "team"),
    ("doi tuyen quoc gia", "team"),
    ("doi tuyen", "team"),
    ("doi bong", "team"),
    ("cau thu", "player"),
    ("thu mon", "player"),
    ("tien dao", "player"),
    ("hau ve", "player"),
    ("tien ve", "player"),
    ("trong tai", "referee"),
    ("san van dong", "venue"),
    ("doi", "team"),
    ("san", "venue"),
    # English
    ("football club", "team"),
    ("soccer club", "team"),
    ("national team", "team"),
    ("club", "team"),
    ("team", "team"),
    ("manager", "player"),
    ("coach", "player"),
    ("head coach", "player"),
    ("player", "player"),
    ("goalkeeper", "player"),
    ("striker", "player"),
    ("defender", "player"),
    ("midfielder", "player"),
    ("winger", "player"),
    ("referee", "referee"),
    ("stadium", "venue"),
    ("arena", "venue"),
    ("ground", "venue"),
]


def _detect_entity_type(entity_name: str, query: str) -> str:
    """Return SoccerEntities field name by looking for type labels before the entity name in query."""
    q = query.lower()
    name = entity_name.lower()
    for label, entity_type in _QUERY_TYPE_LABELS:
        if f"{label} {name}" in q:
            return entity_type
    return "unknown"


_tavily_service: Optional[TavilyService] = None


def _get_tavily() -> TavilyService:
    global _tavily_service
    if _tavily_service is None:
        _tavily_service = TavilyService()
    return _tavily_service


class _DBAnswer(BaseModel):
    answer: str = Field(description="Answer synthesized from the entity data")
    has_sufficient_info: bool = Field(
        description=(
            "True only when every entity was found AND data fully answers the query. "
            "False if any entity is NOT FOUND, data is incomplete, or answer is uncertain."
        )
    )


class _WebAnswer(BaseModel):
    answer: str = Field(description="Answer synthesized from the web content")
    entity_types: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "For each entity name that was NOT found in the local DB, classify its type. "
            "Keys are entity names; values are one of: 'player', 'team', 'venue', 'referee'. "
            "Only include entities that were missing from DB. Leave empty if all were found."
        ),
    )


class EntityAugmentInput(BaseModel):
    entity_names: List[str] = Field(
        ...,
        description=(
            "List of soccer-related entity names to look up. Rules:\n"
            "1. Use the core name only — DO NOT include organizational suffixes or prefixes such as FC, AFC, CF, SC, SSC, AC, EC, FK, BK. "
            "Write 'Manchester City' not 'Manchester City FC', 'Barcelona' not 'FC Barcelona', 'Real Madrid' not 'Real Madrid CF'.\n"
            "2. Each item must be a plain name with no extra narration or type labels.\n"
            "3. Capitalize the first letter of each word.\n"
            "4. Do not pass raw user questions — only resolved entity names."
        ),
        examples=[
            ["Lionel Messi", "Barcelona"],
            ["Kylian Mbappe", "Parc des Princes"],
            ["Manchester City"],
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
    Given a list of entity names and a text query, the tool looks up the entities in the local
    soccer knowledge base and synthesizes the relevant information into a final answer in one step.
    Always use it for background information on players, teams, coaches, referees, venues, etc. —
    including biography, career history, achievements, awards, trophies, season-level stats,
    transfer history, stadium facts, and entity comparisons. Entity facts must come from the DB,
    not from background knowledge.
    """
    args_schema: Type[BaseModel] = EntityAugmentInput  # type: ignore
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    _llm: Any = PrivateAttr()

    def __init__(self):
        super().__init__(description=tool_description("entity_augment"))
        self._llm = get_llm("retrieval-augment")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _run(
        self,
        entity_names: List[str],
        query: str,
        _run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, SearchingResult]:
        return asyncio.run(self._arun(entity_names, query))

    async def _arun(
        self,
        entity_names: List[str],
        query: str,
        _run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, SearchingResult]:
        run_tree = get_current_run_tree()
        try:
            if not entity_names:
                logger.info("No entity names provided to entity_augment.")
                return (
                    "No entity names were provided. Please supply at least one entity name.",
                    SearchingResult(),
                )

            buckets: dict[str, list[str]] = {f: [] for f in SoccerEntities.model_fields}
            for n in entity_names:
                buckets[_detect_entity_type(n, query)].append(n)
            entities = SoccerEntities(**{k: (v if v else None) for k, v in buckets.items()})
            logger.info(f"Received entities for lookup: {entities}")

            searching_result = EntityAugmentTool._query_database(entities)
            logger.info(
                f"DB lookup: found={len(searching_result.found_entities)}, "
                f"missing={len(searching_result.missing_entities)}"
            )

            # Step 2: LLM structured answer from DB data
            db_answer = await self._generate_answer_from_db(query, searching_result)
            logger.info(f"DB answer sufficient={db_answer.has_sufficient_info}")

            if db_answer.has_sufficient_info:
                logger.info("✅ entity_augment: DB answer sufficient, returning.")
                return db_answer.answer, searching_result

            # Step 3: Tavily fallback
            logger.info("DB answer insufficient — running Tavily fallback.")
            web_answer = await self._tavily_fallback(query, searching_result)
            logger.info(f"✅ entity_augment: web answer={web_answer[:120]}")
            return web_answer, searching_result

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

    # ------------------------------------------------------------------
    # Step 2: LLM answer from DB with structured output
    # ------------------------------------------------------------------

    async def _generate_answer_from_db(self, query: str, searching_result: SearchingResult) -> _DBAnswer:
        prompt = get_textual_retrieval_augment_prompt_template()
        structured_llm = self._llm.with_structured_output(_DBAnswer)
        chain = prompt | structured_llm
        searching_text = self._aggregate_searching_results(searching_result)
        try:
            result = await chain.ainvoke({"query": query, "searching_result": searching_text})
            if isinstance(result, _DBAnswer):
                return result
            # Unexpected output type — treat as insufficient
            logger.warning(f"Unexpected structured output type: {type(result)}")
            return _DBAnswer(answer=str(result), has_sufficient_info=False)
        except Exception as e:
            logger.warning(f"Structured output failed ({e}), defaulting to Tavily fallback.")
            return _DBAnswer(answer="", has_sufficient_info=False)

    # ------------------------------------------------------------------
    # Step 3: Tavily fallback
    # ------------------------------------------------------------------

    async def _tavily_fallback(self, query: str, searching_result: SearchingResult) -> str:
        service = _get_tavily()

        # Step 1: Resolve entity
        db_entity = searching_result.found_entities[0] if searching_result.found_entities else None
        name = db_entity.NAME if db_entity else (searching_result.missing_entities[0] if searching_result.missing_entities else None)
        if not name:
            logger.warning("[tavily_fallback] No entity name to resolve — aborting")
            return "Không tìm thấy thông tin bổ sung từ web."

        is_missing = db_entity is None
        logger.info(f"[tavily_fallback] Step 1 — entity='{name}' in_db={not is_missing}")

        # Step 2: Resolve wiki URL
        if db_entity:
            url_field = _ENTITY_URL_FIELD.get(db_entity.ENTITY_TYPE)
            wiki_url: Optional[str] = getattr(db_entity, url_field, None) if url_field else None
            logger.info(f"[tavily_fallback] Step 2 — URL from DB: {wiki_url}")
        else:
            logger.info(f"[tavily_fallback] Step 2 — entity missing from DB, searching wiki URL via Tavily")
            wiki_url = await service.find_wiki_url(name)
            logger.info(f"[tavily_fallback] Step 2 — find_wiki_url result: {wiki_url}")

        # Step 3a: No URL → search_general only, skip DB save
        if not wiki_url:
            logger.info(f"[tavily_fallback] Step 3a — no URL, falling back to search_general (no DB save)")
            fallback = await service.search_general(name)
            if not fallback:
                logger.warning(f"[tavily_fallback] search_general returned empty for '{name}'")
                return "Không tìm thấy thông tin bổ sung từ web."
            logger.info(f"[tavily_fallback] search_general returned {len(fallback)} results")
            web_context = f"## {name}\nSource: web search\n" + "\n".join(
                r.get("content", "") for r in fallback[:2] if r.get("content")
            )
            return await self._generate_web_answer(query, web_context)

        # Step 3b: Extract wiki page
        logger.info(f"[tavily_fallback] Step 3b — extracting wiki: {wiki_url}")
        extract_result = await service.extract_wiki(wiki_url)
        if extract_result and extract_result.markdown:
            cleaned = clean_wiki_markdown(extract_result.markdown)
            logger.info(f"[tavily_fallback] Step 3b — extract OK, cleaned_len={len(cleaned)}, images={len(extract_result.images)}")
            web_context = f"## {name}\nSource: {wiki_url}\n{cleaned}"
        else:
            logger.warning(f"[tavily_fallback] Step 3b — extract failed, falling back to search_general (no DB save)")
            fallback = await service.search_general(name)
            if not fallback:
                logger.warning(f"[tavily_fallback] search_general also returned empty for '{name}'")
                return "Không tìm thấy thông tin bổ sung từ web."
            logger.info(f"[tavily_fallback] search_general returned {len(fallback)} results")
            web_context = f"## {name}\nSource: web search\n" + "\n".join(
                r.get("content", "") for r in fallback[:2] if r.get("content")
            )
            return await self._generate_web_answer(query, web_context)

        # Step 4: Generate answer + classify entity_type for new entities (D14)
        # Inject a note so the LLM knows which entity is missing and must be classified.
        llm_context = web_context
        if is_missing:
            llm_context = (
                f"[CLASSIFICATION REQUIRED: '{name}' is NOT in the local database. "
                f"You MUST fill entity_types[\"{name}\"] with one of: player, team, venue, referee.]\n\n"
                + web_context
            )
        logger.info(f"[tavily_fallback] Step 4 — generating answer via LLM (is_missing={is_missing})")
        prompt = get_textual_retrieval_augment_prompt_template()
        web_llm = self._llm.with_structured_output(_WebAnswer)
        try:
            web_result = await (prompt | web_llm).ainvoke({"query": query, "searching_result": llm_context})
            answer = web_result.answer if isinstance(web_result, _WebAnswer) else str(web_result)
            entity_type = web_result.entity_types.get(name, "").lower() if isinstance(web_result, _WebAnswer) else ""
            logger.info(f"[tavily_fallback] Step 4 — answer_len={len(answer)}, entity_type='{entity_type}'")
        except Exception as e:
            logger.warning(f"[tavily_fallback] Step 4 — LLM failed: {e}")
            answer = web_context[:2000]
            entity_type = ""

        # Step 5: Background upsert MongoDB (don't block response)
        logger.info(f"[tavily_fallback] Step 5 — scheduling background upsert (is_missing={is_missing}, entity_type='{entity_type}')")
        asyncio.create_task(
            self._background_upsert(
                name=name,
                db_entity=db_entity,
                is_missing=is_missing,
                wiki_url=wiki_url,
                cleaned=cleaned,
                images=extract_result.images,
                entity_type=entity_type,
            )
        )

        return answer

    async def _generate_web_answer(self, query: str, web_context: str) -> str:
        """Generate answer from web context without DB save (search_general path)."""
        prompt = get_textual_retrieval_augment_prompt_template()
        web_llm = self._llm.with_structured_output(_WebAnswer)
        try:
            result = await (prompt | web_llm).ainvoke({"query": query, "searching_result": web_context})
            return result.answer if isinstance(result, _WebAnswer) else str(result)
        except Exception as e:
            logger.warning(f"Web answer generation failed: {e}")
            return web_context[:2000]

    # ------------------------------------------------------------------
    # Background DB upsert after Tavily wiki extract
    # ------------------------------------------------------------------

    async def _background_upsert(
        self,
        name: str,
        db_entity: Optional[Any],
        is_missing: bool,
        wiki_url: str,
        cleaned: str,
        images: List[str],
        entity_type: str,
    ) -> None:
        """Force-replace (found entity) or insert (missing entity) in MongoDB."""
        try:
            import main
            client = main.mongo_client if hasattr(main, "mongo_client") and main.mongo_client else None
        except (ImportError, AttributeError):
            client = None

        if client is None:
            resolver.default_resolver = resolver.Resolver(configure=False)
            resolver.default_resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
            client = pymongo.MongoClient(settings.MONGO_SRV, server_api=ServerApi("1"))

        collection = client[settings.SOCCER_DB_NAME][settings.SOCCER_COLLECTION_NAME]

        try:
            if not is_missing and db_entity is not None:
                # Force-replace: rebuild from existing entity, overwrite content fields
                doc = {k: v for k, v in db_entity.model_dump().items() if k != "_id" and v is not None}
                doc.pop("INFOBOX", None)
                doc["SUMMARY"] = extract_summary(cleaned)
                doc["CONTENT"] = cleaned
                doc["IMAGES"] = images
                collection.replace_one({"NAME": {"$regex": f"^{name}$", "$options": "i"}}, doc)
                logger.info(f"[upsert] Force-replaced DB entity: {name}")
            else:
                # Insert new entity (missing from DB, wiki URL found)
                if entity_type not in _ENTITY_URL_FIELD:
                    logger.warning(f"[upsert] Unknown entity_type '{entity_type}' for '{name}', skipping insert")
                    return
                url_field = _ENTITY_URL_FIELD[entity_type]
                doc = {
                    "NAME": name,
                    "ENTITY_TYPE": entity_type,
                    "SUMMARY": extract_summary(cleaned),
                    "CONTENT": cleaned,
                    "IMAGES": images,
                    url_field: wiki_url,
                }
                collection.insert_one(doc)
                logger.info(f"[upsert] Inserted new DB entity: {name} ({entity_type})")
        except Exception as e:
            logger.error(f"[upsert] Background DB upsert failed for '{name}': {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_searching_results(searching_result: SearchingResult) -> str:
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
                    name_conditions = [
                        {"NAME": {"$regex": v, "$options": "i"}}
                        for v in _name_regex_variants(entity_name)
                    ]
                    name_filter = {"$or": name_conditions} if len(name_conditions) > 1 else name_conditions[0]
                    if entity_type == "unknown":
                        filter_q = name_filter
                    else:
                        filter_q = {"$and": [{"ENTITY_TYPE": entity_type}, name_filter]}

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
