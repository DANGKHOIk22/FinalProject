import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Type, Optional, Literal, Annotated, Union, Any, Tuple

from app.soccer_agent.toolbox._config_loader import tool_description

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import InjectedState, BaseTool
from langsmith import get_current_run_tree
from pymongo import MongoClient
from pymongo.collection import Collection
from dns import resolver

from app.config.config import GAME_FALLBACK_TOP_K
from app.config.settings import settings
from app.schema.match import Annotation, MatchInfo
from app.soccer_agent.factory.llm_provider import get_llm
from app.soccer_agent.prompts.toolbox.game_retrieval import (
    get_game_info_retrieval_prompt_template,
    get_game_history_retrieval_prompt_template,
)
from app.soccer_agent.prompts.toolbox.game_search import (
    get_extraction_prompt_template,
    get_match_selection_prompt_template,
)
from app.soccer_agent.services.tavily_service import TavilyService

_tavily: Optional[TavilyService] = None

def _get_tavily() -> TavilyService:
    global _tavily
    if _tavily is None:
        _tavily = TavilyService()
    return _tavily


def _format_news_fallback(results: list, top_k: int, answer: Optional[str] = None) -> str:
    """Format Tavily search results into a readable string, sorted by score."""
    items = sorted(results, key=lambda x: x.get("score") or 0.0, reverse=True)[:top_k]
    if not items and not answer:
        return "No relevant match information found via web search."
    lines = ["[Web search fallback — match not found in local database]"]
    if answer:
        lines.append(f"\nAnswer: {answer}")
    for i, r in enumerate(items, 1):
        title = r.get("title", "N/A")
        content = r.get("content", "").strip()[:600]
        lines.append(f"\n--- Result {i} ---")
        lines.append(f"Title: {title}")
        lines.append(content)
    return "\n".join(lines)



logger = logging.getLogger(__name__)


# ==========================================
# Shared schemas
# ==========================================
class GameQueryInput(BaseModel):
    query: str = Field(description="Câu hỏi hoặc truy vấn của người dùng về trận đấu.")
    time_context: Optional[str] = Field(default=None, description="Bối cảnh thời gian hiện tại.")
    about_current_game: bool = Field(
        default=False,
        description=(
            "Set True only when this sub-query is about the SAME fixture the user is currently watching — "
            "either deictic ('what just happened', 'who has the ball', 'the current score', 'this match'/'this game') "
            "or naming the same teams AND matching the watched game's season/date. "
            "Set False when the query names a different fixture, including the same teams in another season or on another date; "
            "when unsure, set False so the tool resolves by search."
        ),
    )
    execution_agent_state: Annotated[dict, InjectedState] = Field(
        description="Injected worker state — provides game_id for the active HLS session."
    )


class GameHistoryInput(BaseModel):
    query: str = Field(description="Câu hỏi hoặc truy vấn của người dùng về trận đấu.")
    time_context: Optional[str] = Field(default=None, description="Bối cảnh thời gian hiện tại.")
    about_current_game: bool = Field(
        default=False,
        description=(
            "Set True only when this sub-query is about the SAME fixture the user is currently watching — "
            "either deictic ('what just happened', 'who scored', 'this match'/'this game') "
            "or naming the same teams AND matching the watched game's season/date. "
            "Set False when the query names a different fixture, including the same teams in another season or on another date; "
            "when unsure, set False so the tool resolves by search."
        ),
    )
    execution_agent_state: Annotated[dict, InjectedState] = Field(
        description="Trạng thái hiện tại của execution agent. Nếu tool trước là commentary_generation, artifact sẽ là List[Annotation]."
    )


class ToolOutput(BaseModel):
    answer: str = Field(description="Câu trả lời dựa trên thông tin được truy xuất.")


class _FinalResult(BaseModel):
    game_id: Optional[str] = Field(default=None)
    response_llm: str


# ==========================================
# Discriminated result type for game search
# ==========================================
@dataclass
class _GameFound:
    message: str
    game_id: str


@dataclass
class _GameNotFound:
    reason: str


@dataclass
class _GameSearchError:
    detail: str


_FindResult = Union[_GameFound, _GameNotFound, _GameSearchError]


# ==========================================
# Module-level MongoDB singleton
# — one connection pool shared across both tools
# ==========================================
_mongo_client: Optional[MongoClient] = None


def _get_collection() -> Collection:
    global _mongo_client
    if _mongo_client is None:
        _r = resolver.Resolver(configure=False)
        _r.nameservers = ['8.8.8.8', '1.1.1.1']
        resolver.default_resolver = _r
        _mongo_client = MongoClient(settings.MONGO_SRV)
    return _mongo_client[settings.SOCCER_DB_NAME][settings.GAME_COLLECTION_NAME]


# ==========================================
# Shared game-search helper (not a tool)
# ==========================================
class _GameFinder:
    """Reusable search logic shared by both retrieval tools."""

    def __init__(self, llm: Any, collection: Collection) -> None:
        self._llm = llm
        self._collection = collection
        self._parser = PydanticOutputParser(pydantic_object=MatchInfo)
        self.last_time_range: str = "month"

    def find(self, query: str, time_context: Optional[str] = None) -> _FindResult:
        """
        Return a discriminated result:
          _GameFound     — match located, game_id available
          _GameNotFound  — query valid but no match in DB (or LLM uncertain)
          _GameSearchError — unexpected failure (network, LLM, parse error)
        """
        try:
            info = self._extract(query, time_context)
            self.last_time_range = info.time_range
            candidates = self._candidates(info)
            return self._select(candidates, info, query)
        except Exception as e:
            logger.error(f"_GameFinder error: {e}", exc_info=True)
            return _GameSearchError(detail=str(e))

    def _extract(self, query: str, time_context: Optional[str] = None) -> MatchInfo:
        prompt = get_extraction_prompt_template()
        return (prompt | self._llm | self._parser).invoke({
            "question": query,
            "time_context": time_context or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "format_instructions": self._parser.get_format_instructions(),
        })

    def _build_date_filter(self, info: MatchInfo) -> Optional[dict]:
        """Build a single $regex that combines year and month when both are known."""
        has_year = info.year != "unknown"
        has_month = info.month != "unknown"
        if has_year and has_month:
            m = info.month.lstrip("0").zfill(2)
            return {"$regex": f"^{info.year}-{m}"}
        if has_year:
            return {"$regex": f"^{info.year}"}
        if has_month:
            m = info.month.lstrip("0").zfill(2)
            return {"$regex": f"^\\d{{4}}-{m}"}
        return None

    def _candidates(self, info: MatchInfo) -> list[dict]:
        f: dict = {}
        if info.league != "unknown":
            f["league"] = info.league
        if info.season != "unknown":
            f["season"] = info.season

        date_filter = self._build_date_filter(info)
        if date_filter:
            f["date"] = date_filter

        t1 = info.team1 if info.team1 != "unknown" else ""
        t2 = info.team2 if info.team2 != "unknown" else ""
        if t1 and t2:
            f["$or"] = [
                {"home_team": {"$regex": t1, "$options": "i"}, "away_team": {"$regex": t2, "$options": "i"}},
                {"home_team": {"$regex": t2, "$options": "i"}, "away_team": {"$regex": t1, "$options": "i"}},
            ]
        elif t1:
            f["$or"] = [{"home_team": {"$regex": t1, "$options": "i"}}, {"away_team": {"$regex": t1, "$options": "i"}}]
        elif t2:
            f["$or"] = [{"home_team": {"$regex": t2, "$options": "i"}}, {"away_team": {"$regex": t2, "$options": "i"}}]

        projection = {"game_id": 1, "league": 1, "season": 1, "date": 1, "home_team": 1, "away_team": 1, "score": 1, "_id": 0}
        return list(self._collection.find(f, projection).limit(20))

    def _select(self, candidates: list[dict], info: MatchInfo, question: str) -> _FindResult:
        if not candidates:
            logger.warning(f"🔍 GameFinder: no candidates found for query. Extracted info: {info}")
            return _GameNotFound(
                reason="We did not find the match you mentioned in the database. "
                       "Stop the execution and ask user give more specific information."
            )
        if len(candidates) == 1:
            g = candidates[0]
            logger.info(f"✅ GameFinder: single candidate — game_id={g['game_id']} | {g['home_team']} vs {g['away_team']} ({g['date']})")
            return _GameFound(
                message=f"Found match: {g['home_team']} vs {g['away_team']} ({g['date']}).",
                game_id=g["game_id"],
            )

        candidate_text = ""
        for i, g in enumerate(candidates, 1):
            candidate_text += (
                f"\nCandidate {i}:\n"
                f"  - Date: {g['date']}, League: {g['league']}\n"
                f"  - Match: {g['home_team']} vs {g['away_team']}\n"
                f"  - Score: {g['score']}, game_id: {g['game_id']}\n"
            )
        structured_llm = self._llm.with_structured_output(_FinalResult)
        response: _FinalResult = (get_match_selection_prompt_template() | structured_llm).invoke({
            "question": question,
            "info": info.model_dump_json(),
            "candidates": candidate_text,
        })  # type: ignore
        if response.game_id:
            logger.info(f"✅ GameFinder: LLM selected game_id={response.game_id} from {len(candidates)} candidates")
            return _GameFound(message=response.response_llm, game_id=response.game_id)
        logger.warning(f"🔍 GameFinder: LLM could not select a match from {len(candidates)} candidates")
        return _GameNotFound(reason=response.response_llm)


# ==========================================
# Tool: Game Info Retrieval
# ==========================================
class GameInfoRetrievalTool(BaseTool):
    name: str = "game_info_retrieval"
    description: str = """
    Retrieve structured match metadata and final results directly from the match database.
    Handles both the search and retrieval in one step — no prior game_search call needed.
    Returns referee, coaches, attendance, formation, venue, lineups, and final score.
    Use for questions about static pre- or post-match facts.
    """
    args_schema: Type[BaseModel] = GameQueryInput

    _llm: Any = PrivateAttr()
    _collection: Any = PrivateAttr()
    _finder: Any = PrivateAttr()
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    def __init__(self) -> None:
        super().__init__(description=tool_description("game_info_retrieval"))
        self._llm = get_llm("retrieval-augment")
        self._collection = _get_collection()
        self._finder = _GameFinder(self._llm, self._collection)

    async def warmup(self) -> None:
        import asyncio
        from app.soccer_agent.factory.llm_provider import warm_llm
        extraction_msg = get_extraction_prompt_template().messages[0]
        selection_msg = get_match_selection_prompt_template().messages[0]
        await asyncio.gather(
            warm_llm(self._llm, extraction_msg, "game_info_retrieval/extraction"),
            warm_llm(self._llm, selection_msg, "game_info_retrieval/selection"),
        )

    def _fetch_metadata(self, game_id: str) -> str:
        doc = self._collection.find_one({"game_id": game_id}, {"raw": 1, "_id": 0})
        if not doc:
            raise RuntimeError(f"Game not found in database: {game_id}")
        data = dict(doc.get("raw", {}))
        data.pop("annotations", None)
        data.pop("comments", None)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _answer_from_context(self, query: str, context: str, time_context: Optional[str]) -> str:
        llm_structured = self._llm.with_structured_output(ToolOutput)
        response: ToolOutput = (get_game_info_retrieval_prompt_template() | llm_structured).invoke({
            "query": query,
            "context": context,
            "time_context": time_context or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        })  # type: ignore
        logger.info(f"✅ game_info_retrieval | answer={response.answer[:200]}")
        return response.answer

    async def _answer_from_context_async(self, query: str, context: str, time_context: Optional[str]) -> str:
        # Async twin of _answer_from_context — a sync .invoke() here would block
        # the event loop for the whole LLM call and serialize parallel workers.
        llm_structured = self._llm.with_structured_output(ToolOutput)
        response: ToolOutput = await (get_game_info_retrieval_prompt_template() | llm_structured).ainvoke({
            "query": query,
            "context": context,
            "time_context": time_context or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        })  # type: ignore
        logger.info(f"✅ game_info_retrieval | answer={response.answer[:200]}")
        return response.answer

    def _run(
        self,
        query: str,
        execution_agent_state: Annotated[dict, InjectedState],
        about_current_game: bool = False,
        time_context: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Optional[str]]:
        run_tree = get_current_run_tree()
        try:
            game_id = (execution_agent_state.get("additional_material") or {}).get("game_id")

            # Fast path: sub-query is about the currently-playing video — skip search entirely.
            if about_current_game and game_id:
                logger.info(f"⚡ game_info_retrieval fast path — active video game_id={game_id}")
                return self._answer_from_context(query, self._fetch_metadata(game_id), time_context), game_id

            logger.info(f"🔎 GameInfoRetrieval searching: {query}")
            result = self._finder.find(query, time_context)

            if isinstance(result, _GameSearchError):
                error_msg = f"Error in game_info_retrieval search: {result.detail}"
                logger.error(error_msg)
                if run_tree:
                    run_tree.end(error=error_msg)
                return f"An error occurred while searching for the game. Details: {result.detail}. Please try again or stop the execution.", None

            if isinstance(result, _GameNotFound):
                # Safety net: an active video is a strong signal — degrade to it instead of failing.
                if game_id:
                    logger.info(f"↩️ game_info_retrieval search miss — falling back to active video game_id={game_id}")
                    return self._answer_from_context(query, self._fetch_metadata(game_id), time_context), game_id
                return result.reason, None

            # _GameFound
            logger.info(f"📄 Fetching metadata for game_id: {result.game_id}")
            return self._answer_from_context(query, self._fetch_metadata(result.game_id), time_context), result.game_id

        except Exception as e:
            error_msg = f"Error in game_info_retrieval: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if run_tree:
                run_tree.end(error=error_msg)
            return f"An error occurred while retrieving game info. Details: {str(e)}. Please try again or stop the execution.", None

    async def _arun(
        self,
        query: str,
        execution_agent_state: Annotated[dict, InjectedState],
        about_current_game: bool = False,
        time_context: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Optional[str]]:
        """Async version. Fast path for the active video; Tavily fallback when no match is found and no video is active."""
        run_tree = get_current_run_tree()
        try:
            game_id = (execution_agent_state.get("additional_material") or {}).get("game_id")

            # Fast path: sub-query is about the currently-playing video — skip search entirely.
            if about_current_game and game_id:
                logger.info(f"⚡ game_info_retrieval (async) fast path — active video game_id={game_id}")
                return await self._answer_from_context_async(query, self._fetch_metadata(game_id), time_context), game_id

            logger.info(f"🔎 GameInfoRetrieval (async) searching: {query}")
            result = self._finder.find(query, time_context)

            if isinstance(result, _GameSearchError):
                error_msg = f"Error in game_info_retrieval search: {result.detail}"
                logger.error(error_msg)
                if run_tree:
                    run_tree.end(error=error_msg)
                return f"An error occurred while searching for the game: {result.detail}", None

            if isinstance(result, _GameNotFound):
                # Safety net: an active video is a strong signal — prefer it over a web search.
                if game_id:
                    logger.info(f"↩️ game_info_retrieval (async) search miss — falling back to active video game_id={game_id}")
                    return await self._answer_from_context_async(query, self._fetch_metadata(game_id), time_context), game_id
                logger.info(f"⚡ game_info_retrieval DB miss — falling back to Tavily search_combine")
                tavily_answer, news = await _get_tavily().search_combine(
                    f"{query} match result score lineup",
                    time_range=self._finder.last_time_range,
                    max_results=5,
                    search_depth="fast"
                )
                news_text = _format_news_fallback(news, GAME_FALLBACK_TOP_K, answer=tavily_answer)
                llm_structured = self._llm.with_structured_output(ToolOutput)
                response: ToolOutput = await (get_game_info_retrieval_prompt_template() | llm_structured).ainvoke({  # type: ignore
                    "query": query,
                    "context": news_text,
                    "time_context": time_context or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                })
                return response.answer, news_text

            # _GameFound
            logger.info(f"📄 Fetching metadata for game_id: {result.game_id}")
            return await self._answer_from_context_async(query, self._fetch_metadata(result.game_id), time_context), result.game_id

        except Exception as e:
            error_msg = f"Error in game_info_retrieval: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if run_tree:
                run_tree.end(error=error_msg)
            return f"An error occurred while retrieving game info: {str(e)}", None


# ==========================================
# Tool: Game History Retrieval
# ==========================================
class GameHistoryRetrievalTool(BaseTool):
    name: str = "game_history_retrieval"
    description: str = """
    Load the match event timeline or commentary and generate answers grounded in those events.
    Handles both the search and retrieval in one step — no prior game_search call needed.
    If the previous tool was commentary_generation, its Annotation artifact is used directly instead.
    Use for questions about specific events, timestamps, substitutions, goals, and cards.
    """
    args_schema: Type[BaseModel] = GameHistoryInput

    _llm: Any = PrivateAttr()
    _collection: Any = PrivateAttr()
    _finder: Any = PrivateAttr()
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    def __init__(self) -> None:
        super().__init__(description=tool_description("game_history_retrieval"))
        self._llm = get_llm("retrieval-augment")
        self._collection = _get_collection()
        self._finder = _GameFinder(self._llm, self._collection)

    async def warmup(self) -> None:
        import asyncio
        from app.soccer_agent.factory.llm_provider import warm_llm
        extraction_msg = get_extraction_prompt_template().messages[0]
        selection_msg = get_match_selection_prompt_template().messages[0]
        await asyncio.gather(
            warm_llm(self._llm, extraction_msg, "game_history_retrieval/extraction"),
            warm_llm(self._llm, selection_msg, "game_history_retrieval/selection"),
        )

    def _history_from_game_id(self, game_id: str, active_vct: Optional[float] = None) -> str:
        doc = self._collection.find_one({"game_id": game_id}, {"raw": 1, "_id": 0})
        if not doc:
            raise RuntimeError(f"Game not found in database: {game_id}")

        data = doc.get("raw", {})
        processed: List[Annotation] = []

        if "annotations" in data:
            for event in data.get("annotations", []):
                timestamp = event.get("contrastive_aligned_gameTime", "") or event.get("gameTime", "")
                processed.append(Annotation(
                    description=event.get("description", ""),
                    label=event.get("label", "unknown"),
                    gameTime=timestamp,
                ))
        elif "comments" in data:
            for comment in data.get("comments", []):
                t = comment.get("time_stamp")
                if active_vct is not None and isinstance(t, (int, float)) and t > active_vct:
                    continue
                processed.append(Annotation(
                    description=comment.get("comments_text", ""),
                    label=comment.get("comments_type", "unknown"),
                    gameTime=f"{comment.get('half', '')} - {comment.get('time_stamp', '')}",
                ))
        else:
            raise ValueError("Match must contain 'annotations' or 'comments' key.")

        if not processed:
            raise ValueError(f"Match history contains no events for game_id={game_id}.")

        return json.dumps([a.model_dump() for a in processed], indent=2, ensure_ascii=False)

    @staticmethod
    def _format_position(seconds: float) -> str:
        s = int(seconds)
        return f"{s // 60}:{s % 60:02d} ({s}s)"  # e.g. "45:00 (2700s)"

    def _resolve_history_context(
        self,
        query: str,
        last_artifact: Union[List[Annotation], str, None],
        active_game_id: Optional[str] = None,
        about_current_game: bool = False,
        active_vct: Optional[float] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Return (history_json, game_id).
        game_id is None when history comes from commentary_generation annotations.

        Resolution precedence:
          1. last_tool_artifact from a previous tool in the chain
          2. the currently-playing video (when about_current_game is set)
          3. search from the query (with the active video as a safety net on miss)
        """
        if isinstance(last_artifact, list):
            # From commentary_generation — List[Annotation], use directly
            annotations = [a.model_dump() if hasattr(a, "model_dump") else a for a in last_artifact]
            return json.dumps(annotations, indent=2, ensure_ascii=False), None

        if isinstance(last_artifact, str) and last_artifact:
            # Validate it looks like a game_id (contains path separators typical of our IDs)
            if "/" in last_artifact:
                return self._history_from_game_id(last_artifact), last_artifact
            # Doesn't look like a game_id — fall through
            logger.warning(f"last_tool_artifact '{last_artifact[:80]}' does not look like a game_id, searching instead.")

        # Fast path: sub-query is about the currently-playing video — skip search entirely.
        if about_current_game and active_game_id:
            logger.info(f"⚡ game_history_retrieval fast path — active video game_id={active_game_id}")
            return self._history_from_game_id(active_game_id, active_vct=active_vct), active_game_id

        # Search from query
        result = self._finder.find(query)
        if isinstance(result, _GameSearchError):
            raise RuntimeError(f"Game search error: {result.detail}")
        if isinstance(result, _GameNotFound):
            # Safety net: an active video is a strong signal — degrade to it instead of failing.
            if active_game_id:
                logger.info(f"↩️ game_history_retrieval search miss — falling back to active video game_id={active_game_id}")
                return self._history_from_game_id(active_game_id, active_vct=active_vct), active_game_id
            raise ValueError(result.reason)
        return self._history_from_game_id(result.game_id), result.game_id

    def _run(
        self,
        query: str,
        execution_agent_state: Annotated[dict, InjectedState],
        about_current_game: bool = False,
        time_context: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Optional[str]]:
        run_tree = get_current_run_tree()
        try:
            last_artifact = execution_agent_state.get("last_tool_artifact")
            active_game_id = (execution_agent_state.get("additional_material") or {}).get("game_id")
            active_vct = execution_agent_state.get("video_current_time")
            artifact_preview = (
                last_artifact if isinstance(last_artifact, str)
                else f"[{len(last_artifact)} annotations]" if isinstance(last_artifact, list)
                else "None — will search"
            )
            logger.info(f"📖 GameHistoryRetrieval artifact: {artifact_preview}")

            on_active = active_game_id is not None
            history_context, game_id = self._resolve_history_context(
                query, last_artifact, active_game_id, about_current_game,
                active_vct=active_vct if on_active else None,
            )

            # Ground the answer on the live playback position only when answering about the active video.
            on_active = game_id is not None and game_id == active_game_id
            video_position = self._format_position(active_vct) if (on_active and active_vct is not None) else "None"

            llm_structured = self._llm.with_structured_output(ToolOutput)
            response: ToolOutput = (get_game_history_retrieval_prompt_template() | llm_structured).invoke({
                "query": query,
                "context": history_context,
                "time_context": time_context or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "video_position": video_position,
            })  # type: ignore
            logger.info(f"✅ game_history_retrieval: game_id={game_id} | on_active={on_active} | answer={response.answer[:200]}")
            return response.answer, game_id

        except Exception as e:
            error_msg = f"Error in game_history_retrieval: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if run_tree:
                run_tree.end(error=error_msg)
            return f"An error occurred while retrieving game history. Details: {str(e)}. Please try again or stop the execution.", None

    async def _arun(
        self,
        query: str,
        execution_agent_state: Annotated[dict, InjectedState],
        about_current_game: bool = False,
        time_context: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Optional[str]]:
        """Async version with Tavily fallback (match report) when match not found in DB."""
        run_tree = get_current_run_tree()
        try:
            last_artifact = execution_agent_state.get("last_tool_artifact")
            active_game_id = (execution_agent_state.get("additional_material") or {}).get("game_id")
            active_vct = execution_agent_state.get("video_current_time")
            artifact_preview = (
                last_artifact if isinstance(last_artifact, str)
                else f"[{len(last_artifact)} annotations]" if isinstance(last_artifact, list)
                else "None — will search"
            )
            logger.info(f"📖 GameHistoryRetrieval (async) artifact: {artifact_preview}")

            try:
                on_active = active_game_id is not None
                history_context, game_id = self._resolve_history_context(
                    query, last_artifact, active_game_id, about_current_game,
                    active_vct=active_vct if on_active else None,
                )
            except ValueError:
                # _GameNotFound path — fallback to Tavily match report search (no active video to ground on)
                logger.info("⚡ game_history_retrieval DB miss — falling back to Tavily search_combine")
                tavily_answer, news = await _get_tavily().search_combine(
                    f"{query} match report events goals cards substitutions",
                    time_range=self._finder.last_time_range,
                    max_results=5,
                    search_depth="fast",
                )
                news_text = _format_news_fallback(news, GAME_FALLBACK_TOP_K, answer=tavily_answer)
                llm_structured = self._llm.with_structured_output(ToolOutput)
                response: ToolOutput = await (get_game_history_retrieval_prompt_template() | llm_structured).ainvoke({  # type: ignore
                    "query": query,
                    "context": news_text,
                    "time_context": time_context or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "video_position": "None",
                })
                return response.answer, news_text

            # Ground the answer on the live playback position only when answering about the active video.
            on_active = game_id is not None and game_id == active_game_id
            video_position = self._format_position(active_vct) if (on_active and active_vct is not None) else "None"

            llm_structured = self._llm.with_structured_output(ToolOutput)
            # ainvoke — a sync .invoke() here blocks the event loop for the whole LLM call
            response: ToolOutput = await (get_game_history_retrieval_prompt_template() | llm_structured).ainvoke({
                "query": query,
                "context": history_context,
                "time_context": time_context or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "video_position": video_position,
            })  # type: ignore
            logger.info(f"✅ game_history_retrieval: game_id={game_id} | on_active={on_active} | answer={response.answer[:200]}")
            return response.answer, history_context

        except Exception as e:
            error_msg = f"Error in game_history_retrieval: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if run_tree:
                run_tree.end(error=error_msg)
            return f"An error occurred while retrieving game history: {str(e)}", None
