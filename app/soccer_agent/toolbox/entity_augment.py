import asyncio
from datetime import datetime, timedelta
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
from app.config.config import ENTITY_FRESHNESS_DAYS
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
from app.soccer_agent.services.content_cleaner import clean_wiki_markdown, extract_summary, strip_summary
from app.soccer_agent.services.tavily_service import TavilyService
from app.soccer_agent.toolbox._config_loader import tool_description
from app.soccer_agent.memory.long_term_memory import long_term_memory_manager
from langfuse import get_client

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
    """Return SoccerEntities field name by looking for type labels in query.

    Checks two patterns:
    - "cầu thủ lionel messi" — label BEFORE name (Vietnamese/English prefix)
    - "lionel messi (player)"  — label/type AFTER name in parentheses
    """
    q = query.lower()
    name = entity_name.lower()
    for label, entity_type in _QUERY_TYPE_LABELS:
        if f"{label} {name}" in q:
            return entity_type
        if f"{name} ({label})" in q:
            return entity_type
    return "unknown"


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse a "%Y-%m-%d %H:%M:%S" timestamp, tolerating a trailing " UTC".

    Returns None when the value is empty or cannot be parsed — callers treat
    that as "unknown freshness" (stale).
    """
    if not value:
        return None
    cleaned = str(value).strip()
    if cleaned.endswith(" UTC"):
        cleaned = cleaned[:-4].strip()
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def _is_fresh(searching_result: "SearchingResult", time_context: Optional[str]) -> bool:
    """True when every found entity's LAST_UPDATED is within ENTITY_FRESHNESS_DAYS.

    Conservative: any found entity missing or with an unparseable LAST_UPDATED
    makes the whole result stale, so we fetch fresh data. Uses the oldest
    LAST_UPDATED among found entities as the reference point.
    """
    if not searching_result.found_entities:
        return False

    now = _parse_dt(time_context) or datetime.now()

    oldest: Optional[datetime] = None
    for entity in searching_result.found_entities:
        parsed = _parse_dt(getattr(entity, "LAST_UPDATED", None))
        if parsed is None:
            return False
        if oldest is None or parsed < oldest:
            oldest = parsed

    return (now - oldest) <= timedelta(days=ENTITY_FRESHNESS_DAYS)


_tavily_service: Optional[TavilyService] = None


def _get_tavily() -> TavilyService:
    global _tavily_service
    if _tavily_service is None:
        _tavily_service = TavilyService()
    return _tavily_service


class _EntityClassification(BaseModel):
    entity_name: str = Field(description="Name of the entity found in the text")
    entity_type: str = Field(description="Type of the entity: 'player', 'team', 'venue', 'referee', or 'unknown'")

class _AugmentedAnswer(BaseModel):
    answer: str = Field(description="Answer synthesized from the provided data (DB or Web content)")
    has_sufficient_info: bool = Field(
        description=(
            "True only when the provided data fully answers the query. "
            "False if data is incomplete, entity is not found, or answer is uncertain."
        )
    )
    unknown_entities: List[_EntityClassification] = Field(
        description=(
            "List of entity classifications for entities that were NOT found in the local database. "
            "Leave empty if all entities were found or if no classification is needed."
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
    time_context: Optional[str] = Field(
        default=None,
        description="Current date and time for temporal reasoning."
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

    async def warmup(self) -> None:
        from app.soccer_agent.factory.llm_provider import warm_llm
        system_msg = get_textual_retrieval_augment_prompt_template().messages[0]
        await warm_llm(self._llm, system_msg, "entity_augment")

    def _run(
        self,
        entity_names: List[str],
        query: str,
        time_context: Optional[str] = None,
        _run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, SearchingResult]:
        return asyncio.run(self._arun(entity_names, query, time_context))

    async def _arun(
        self,
        entity_names: List[str],
        query: str,
        time_context: Optional[str] = None,
        _run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, SearchingResult]:
        if not time_context:
            time_context = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

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

            langfuse = get_client()

            # Step 1: Local DB lookup.
            with langfuse.start_as_current_observation(as_type="chain", name="query_database"):
                searching_result = self._query_database(entities)
                logger.info(
                    f"DB lookup: found={len(searching_result.found_entities)}, "
                    f"missing={len(searching_result.missing_entities)}"
                )

            # Step 2: Nothing in the DB → full Tavily fallback
            # (resolve wiki URL → extract / general search → LLM answer).
            if not searching_result.found_entities:
                logger.info("❌ No entities found in DB — jumping straight to Tavily fallback.")
                with langfuse.start_as_current_observation(as_type="chain", name="tavily_fallback"):
                    web_answer = await self._tavily_fallback(query, searching_result, time_context)
                return web_answer, searching_result

            # Step 3: Freshness gate. When DB data is recent enough, trust it and
            # skip Tavily entirely — even if the LLM flags the answer as insufficient
            # (it tends to second-guess freshness from LAST_UPDATED).
            if _is_fresh(searching_result, time_context):
                logger.info("✅ entity_augment: DB data is fresh — trusting DB answer, skipping Tavily.")
                with langfuse.start_as_current_observation(as_type="chain", name="generate_answer_from_db"):
                    db_answer = await self._generate_answer_from_db(query, searching_result, time_context)
                return db_answer.answer, searching_result

            # Step 4: Stale DB data. Run the DB-answer LLM call and the Tavily wiki
            # fetch concurrently so the fallback adds no latency on the slow path.
            # Both run as child observations under one parent so the trace shows
            # them executing in parallel.
            logger.info("entity_augment: DB data stale — DB answer ∥ Tavily wiki fetch in parallel.")

            async def _db_answer_observed():
                with langfuse.start_as_current_observation(as_type="chain", name="generate_answer_from_db"):
                    return await self._generate_answer_from_db(query, searching_result, time_context)

            async def _tavily_fetch_observed():
                with langfuse.start_as_current_observation(as_type="chain", name="tavily_wiki_fetch"):
                    return await self._fetch_wiki_context(searching_result)

            with langfuse.start_as_current_observation(
                as_type="chain", name="db_answer_parallel_tavily_fetch"
            ):
                db_answer, wiki = await asyncio.gather(
                    _db_answer_observed(),
                    _tavily_fetch_observed(),
                    return_exceptions=True,
                )

            if isinstance(db_answer, BaseException):
                logger.warning(f"DB answer failed on stale path: {db_answer}")
                db_answer = _AugmentedAnswer(answer="", has_sufficient_info=False, unknown_entities=[])
            if isinstance(wiki, BaseException):
                logger.warning(f"Wiki fetch failed on stale path: {wiki}")
                wiki = None

            # Step 5: Pick the final answer — trust the DB answer when sufficient,
            # otherwise regenerate from the freshly fetched wiki markdown.
            if db_answer.has_sufficient_info:
                final_answer = db_answer.answer
            elif wiki:
                logger.info("DB answer insufficient — regenerating from fetched wiki content.")
                with langfuse.start_as_current_observation(as_type="chain", name="generate_web_answer"):
                    final_answer = await self._generate_web_answer(query, wiki["web_context"], time_context)
            else:
                logger.info("DB answer insufficient and wiki fetch unavailable — returning DB answer.")
                final_answer = db_answer.answer

            # Refresh Mongo + LAST_UPDATED in the background whenever we fetched
            # fresh wiki content (both sufficient and insufficient branches).
            # SaveToMemoryNode picks this up and upserts off the main flow.
            if wiki and wiki.get("source") == "wiki_extract":
                searching_result._upsert_payload = {
                    "source": "wiki_extract",
                    "name": wiki["name"],
                    "entity_type": wiki["entity_type"],
                    "is_missing": wiki["is_missing"],
                    "wiki_url": wiki["wiki_url"],
                    "cleaned_content": wiki["cleaned"],
                    "images": wiki["images"],
                }

            return final_answer, searching_result

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

    async def _generate_answer_from_db(self, query: str, searching_result: SearchingResult, time_context: Optional[str] = None) -> _AugmentedAnswer:
        prompt = get_textual_retrieval_augment_prompt_template()
        structured_llm = self._llm.with_structured_output(_AugmentedAnswer)
        chain = prompt | structured_llm
        searching_text = self._aggregate_searching_results(searching_result)
        try:
            result = await chain.ainvoke({"query": query, "searching_result": searching_text, "time_context": time_context or "Unknown"})
            if isinstance(result, _AugmentedAnswer):
                return result
            # Unexpected output type — treat as insufficient
            logger.warning(f"Unexpected structured output type: {type(result)}")
            return _AugmentedAnswer(answer=str(result), has_sufficient_info=False, unknown_entities=[])
        except Exception as e:
            logger.warning(f"Structured output failed ({e}), defaulting to Tavily fallback.")
            return _AugmentedAnswer(answer="", has_sufficient_info=False, unknown_entities=[])

    # ------------------------------------------------------------------
    # Fetch fresh wiki context (no answer generation) — used by the stale
    # path so it can run in parallel with the DB-answer LLM call.
    # ------------------------------------------------------------------

    async def _fetch_wiki_context(self, searching_result: SearchingResult) -> Optional[dict]:
        """Resolve + fetch web/wiki content for the first found entity.

        Fetch only: resolves the wiki URL (from DB), extracts the page, and
        falls back to a general web search. Returns a dict with the LLM-ready
        ``web_context`` plus the metadata needed to build ``_upsert_payload``,
        or None when nothing usable could be fetched. No LLM call is made here.
        """
        service = _get_tavily()

        db_entity = searching_result.found_entities[0] if searching_result.found_entities else None
        if not db_entity or not getattr(db_entity, "NAME", None):
            logger.warning("[fetch_wiki] No found entity to resolve — aborting")
            return None

        name = db_entity.NAME
        entity_type = getattr(db_entity, "ENTITY_TYPE", "unknown").lower() or "unknown"
        url_field = _ENTITY_URL_FIELD.get(db_entity.ENTITY_TYPE)
        wiki_url: Optional[str] = getattr(db_entity, url_field, None) if url_field else None
        logger.info(f"[fetch_wiki] entity='{name}' type='{entity_type}' url={wiki_url}")

        # No URL → general search only
        if not wiki_url:
            _, fallback = await service.search_combine(name, max_results=5, search_depth="fast", time_range="year")
            if not fallback:
                logger.warning(f"[fetch_wiki] search_general empty for '{name}'")
                return None
            web_context = f"## {name}\nSource: web search (general)\n" + "\n".join(
                r.get("content", "") for r in fallback if r.get("content")
            )
            return {
                "web_context": web_context,
                "source": "general_search",
                "name": name,
                "entity_type": entity_type,
                "is_missing": False,
                "wiki_url": "",
                "cleaned": web_context,
                "images": [],
            }

        # Extract wiki page
        extract_result = await service.extract_wiki(wiki_url)
        if extract_result and extract_result.markdown:
            cleaned = clean_wiki_markdown(extract_result.markdown)
            logger.info(f"[fetch_wiki] extract OK, cleaned_len={len(cleaned)}, images={len(extract_result.images)}")
            return {
                "web_context": f"## {name}\nSource: {wiki_url}\n{cleaned}",
                "source": "wiki_extract",
                "name": name,
                "entity_type": entity_type,
                "is_missing": False,
                "wiki_url": wiki_url,
                "cleaned": cleaned,
                "images": extract_result.images,
            }

        # Extract failed → general search fallback
        logger.warning(f"[fetch_wiki] extract failed for {wiki_url}, falling back to search_general")
        _, fallback = await service.search_combine(name, max_results=5, search_depth="fast", time_range="year")
        if not fallback:
            logger.warning(f"[fetch_wiki] search_general also empty for '{name}'")
            return None
        web_context = f"## {name}\nSource: web search\n" + "\n".join(
            r.get("content", "") for r in fallback if r.get("content")
        )
        return {
            "web_context": web_context,
            "source": "general_search",
            "name": name,
            "entity_type": entity_type,
            "is_missing": False,
            "wiki_url": wiki_url,
            "cleaned": web_context,
            "images": [],
        }

    # ------------------------------------------------------------------
    # Step 3: Tavily fallback
    # ------------------------------------------------------------------

    async def _tavily_fallback(self, query: str, searching_result: SearchingResult, time_context: Optional[str] = None) -> str:
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

        # Step 3a: No URL → search_general only
        if not wiki_url:
            logger.info(f"[tavily_fallback] Step 3a — no URL, falling back to search_general")
            _, fallback = await service.search_combine(name, max_results=5, search_depth="fast", time_range="year")
            if not fallback:
                logger.warning(f"[tavily_fallback] search_general returned empty for '{name}'")
                return "Không tìm thấy thông tin bổ sung từ web."

            logger.info(f"[tavily_fallback] search_general returned {len(fallback)} results")
            web_context = f"## {name}\nSource: web search (general)\n" + "\n".join(
                r.get("content", "") for r in fallback if r.get("content")
            )
            # Pack payload for SaveToMemoryNode (general search, no wiki URL)
            resolved_type = getattr(db_entity, "ENTITY_TYPE", "unknown").lower() if db_entity else "unknown"
            searching_result._upsert_payload = {
                "source": "general_search",
                "name": name,
                "entity_type": resolved_type,
                "is_missing": is_missing,
                "wiki_url": "",
                "cleaned_content": web_context,
                "images": [],
            }
            return await self._generate_web_answer(query, web_context, time_context)

        # Step 3b: Extract wiki page
        logger.info(f"[tavily_fallback] Step 3b — extracting wiki: {wiki_url}")
        extract_result = await service.extract_wiki(wiki_url)
        if extract_result and extract_result.markdown:
            cleaned = clean_wiki_markdown(extract_result.markdown)
            logger.info(f"[tavily_fallback] Step 3b — extract OK, cleaned_len={len(cleaned)}, images={len(extract_result.images)}")
            web_context = f"## {name}\nSource: {wiki_url}\n{cleaned}"
        else:
            logger.warning(f"[tavily_fallback] Step 3b — extract failed, falling back to search_general")
            _, fallback = await service.search_combine(name, max_results=5, search_depth="fast", time_range="year")
            if not fallback:
                logger.warning(f"[tavily_fallback] search_general also returned empty for '{name}'")
                return "Không tìm thấy thông tin bổ sung từ web."
            logger.info(f"[tavily_fallback] search_general returned {len(fallback)} results")
            web_context = f"## {name}\nSource: web search\n" + "\n".join(
                r.get("content", "") for r in fallback if r.get("content")
            )
            # Pack payload for SaveToMemoryNode (wiki extract failed, general search fallback)
            resolved_type = getattr(db_entity, "ENTITY_TYPE", "unknown").lower() if db_entity else "unknown"
            searching_result._upsert_payload = {
                "source": "general_search",
                "name": name,
                "entity_type": resolved_type,
                "is_missing": is_missing,
                "wiki_url": wiki_url,
                "cleaned_content": web_context,
                "images": [],
            }
            return await self._generate_web_answer(query, web_context, time_context)

        # Step 4: Generate answer + resolve entity_type
        # Entity type resolution priority:
        #   1. If entity exists in DB → use db_entity.ENTITY_TYPE
        #   2. If missing from DB → ask LLM to classify
        #   3. If LLM cannot determine → "unknown"
        resolved_entity_type = ""
        if not is_missing and db_entity is not None:
            resolved_entity_type = getattr(db_entity, "ENTITY_TYPE", "").lower()
        
        llm_context = f"[LAST_UPDATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n\n" + web_context
        if is_missing:
            llm_context = (
                f"[CLASSIFICATION REQUIRED: '{name}' is NOT in the local database. "
                f"You MUST fill unknown_entities with: {{'entity_name': '{name}', 'entity_type': '<player|team|venue|referee>'}}]\n\n"
                + llm_context
            )
        
        logger.info(f"[tavily_fallback] Step 4 — generating answer via LLM (is_missing={is_missing})")
        prompt = get_textual_retrieval_augment_prompt_template()
        web_llm = self._llm.with_structured_output(_AugmentedAnswer)
        try:
            web_result = await (prompt | web_llm).ainvoke({"query": query, "searching_result": llm_context, "time_context": time_context or "Unknown"})
            answer = web_result.answer if isinstance(web_result, _AugmentedAnswer) else str(web_result)
            
            # Only use LLM's entity_type if we don't already have one from DB
            if not resolved_entity_type and isinstance(web_result, _AugmentedAnswer):
                for ent in web_result.unknown_entities:
                    if ent.entity_name.lower() == name.lower():
                        resolved_entity_type = ent.entity_type.lower()
                        break
        except Exception as e:
            logger.warning(f"[tavily_fallback] Step 4 — LLM failed: {e}")
            answer = web_context[:2000]
        
        # Default to "unknown" if still empty
        if not resolved_entity_type:
            resolved_entity_type = "unknown"
        
        logger.info(f"[tavily_fallback] Step 4 — answer_len={len(answer)}, entity_type='{resolved_entity_type}'")

        # Pack upsert metadata into searching_result for SaveToMemoryNode to process
        searching_result._upsert_payload = {
            "source": "wiki_extract",
            "name": name,
            "entity_type": resolved_entity_type,
            "is_missing": is_missing,
            "wiki_url": wiki_url,
            "cleaned_content": cleaned,
            "images": extract_result.images,
        }

        return answer

    async def _generate_web_answer(self, query: str, web_context: str, time_context: Optional[str] = None) -> str:
        """Generate answer from web context without DB save (search_general path)."""
        prompt = get_textual_retrieval_augment_prompt_template()
        web_llm = self._llm.with_structured_output(_AugmentedAnswer)
        try:
            result = await (prompt | web_llm).ainvoke({"query": query, "searching_result": web_context, "time_context": time_context or "Unknown"})
            return result.answer if isinstance(result, _AugmentedAnswer) else str(result)
        except Exception as e:
            logger.warning(f"Web answer generation failed: {e}")
            return web_context[:2000]



    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_searching_results(searching_result: SearchingResult) -> str:
        aggregated_text = ""
        for entity in searching_result.found_entities:
            last_updated = getattr(entity, "LAST_UPDATED", None)
            last_updated_str = last_updated or "Not available (data may be outdated — treat as potentially stale)"
            aggregated_text += "-" * 10 + "\n"
            aggregated_text += f"INFORMATION ABOUT: {entity.NAME} (ENTITY TYPE: {entity.ENTITY_TYPE}) | LAST_UPDATED: {last_updated_str}\n"
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

    @standard_cache.cache(ttl=60 * 60, validatedModel=SearchingResult)
    def _query_database(self, soccer_entities: SoccerEntities) -> SearchingResult:
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
