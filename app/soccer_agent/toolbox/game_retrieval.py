import json
import logging
from dataclasses import dataclass
from typing import List, Type, Optional, Literal, Annotated, Union, Any, Tuple

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import InjectedState, BaseTool
from langsmith import get_current_run_tree
from pymongo import MongoClient
from pymongo.collection import Collection
from dns import resolver

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

logger = logging.getLogger(__name__)


# ==========================================
# Shared schemas
# ==========================================
class GameQueryInput(BaseModel):
    query: str = Field(description="Câu hỏi hoặc truy vấn của người dùng về trận đấu.")


class GameHistoryInput(BaseModel):
    query: str = Field(description="Câu hỏi hoặc truy vấn của người dùng về trận đấu.")
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

    def find(self, query: str) -> _FindResult:
        """
        Return a discriminated result:
          _GameFound     — match located, game_id available
          _GameNotFound  — query valid but no match in DB (or LLM uncertain)
          _GameSearchError — unexpected failure (network, LLM, parse error)
        """
        try:
            info = self._extract(query)
            candidates = self._candidates(info)
            return self._select(candidates, info, query)
        except Exception as e:
            logger.error(f"_GameFinder error: {e}", exc_info=True)
            return _GameSearchError(detail=str(e))

    def _extract(self, query: str) -> MatchInfo:
        prompt = get_extraction_prompt_template()
        return (prompt | self._llm | self._parser).invoke({
            "question": query,
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
            return _GameNotFound(
                reason="We did not find the match you mentioned in the database. "
                       "Stop the execution and ask user give more specific information."
            )
        if len(candidates) == 1:
            g = candidates[0]
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
            return _GameFound(message=response.response_llm, game_id=response.game_id)
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
        super().__init__()
        self._llm = get_llm("retrieval-augment")
        self._collection = _get_collection()
        self._finder = _GameFinder(self._llm, self._collection)

    def _fetch_metadata(self, game_id: str) -> str:
        doc = self._collection.find_one({"game_id": game_id}, {"raw": 1, "_id": 0})
        if not doc:
            raise RuntimeError(f"Game not found in database: {game_id}")
        data = dict(doc.get("raw", {}))
        data.pop("annotations", None)
        data.pop("comments", None)
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Optional[str]]:
        run_tree = get_current_run_tree()
        try:
            logger.info(f"🔎 GameInfoRetrieval searching: {query}")
            result = self._finder.find(query)

            if isinstance(result, _GameSearchError):
                error_msg = f"Error in game_info_retrieval search: {result.detail}"
                logger.error(error_msg)
                if run_tree:
                    run_tree.end(error=error_msg)
                return f"An error occurred while searching for the game. Details: {result.detail}. Please try again or stop the execution.", None

            if isinstance(result, _GameNotFound):
                return result.reason, None

            # _GameFound
            logger.info(f"📄 Fetching metadata for game_id: {result.game_id}")
            context = self._fetch_metadata(result.game_id)
            llm_structured = self._llm.with_structured_output(ToolOutput)
            response: ToolOutput = (get_game_info_retrieval_prompt_template() | llm_structured).invoke({
                "query": query,
                "context": context,
            })  # type: ignore
            return response.answer, result.game_id

        except Exception as e:
            error_msg = f"Error in game_info_retrieval: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if run_tree:
                run_tree.end(error=error_msg)
            return f"An error occurred while retrieving game info. Details: {str(e)}. Please try again or stop the execution.", None


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
        super().__init__()
        self._llm = get_llm("retrieval-augment")
        self._collection = _get_collection()
        self._finder = _GameFinder(self._llm, self._collection)

    def _history_from_game_id(self, game_id: str) -> str:
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

    def _resolve_history_context(
        self, query: str, last_artifact: Union[List[Annotation], str, None]
    ) -> Tuple[str, Optional[str]]:
        """
        Return (history_json, game_id).
        game_id is None when history comes from commentary_generation annotations.
        """
        if isinstance(last_artifact, list):
            # From commentary_generation — List[Annotation], use directly
            annotations = [a.model_dump() if hasattr(a, "model_dump") else a for a in last_artifact]
            return json.dumps(annotations, indent=2, ensure_ascii=False), None

        if isinstance(last_artifact, str) and last_artifact:
            # Validate it looks like a game_id (contains path separators typical of our IDs)
            if "/" in last_artifact:
                return self._history_from_game_id(last_artifact), last_artifact
            # Doesn't look like a game_id — fall through to search
            logger.warning(f"last_tool_artifact '{last_artifact[:80]}' does not look like a game_id, searching instead.")

        # No valid prior artifact — search from query
        result = self._finder.find(query)
        if isinstance(result, _GameSearchError):
            raise RuntimeError(f"Game search error: {result.detail}")
        if isinstance(result, _GameNotFound):
            raise ValueError(result.reason)
        return self._history_from_game_id(result.game_id), result.game_id

    def _run(
        self,
        query: str,
        execution_agent_state: Annotated[dict, InjectedState],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Optional[str]]:
        run_tree = get_current_run_tree()
        try:
            last_artifact = execution_agent_state.get("last_tool_artifact")
            artifact_preview = (
                last_artifact if isinstance(last_artifact, str)
                else f"[{len(last_artifact)} annotations]" if isinstance(last_artifact, list)
                else "None — will search"
            )
            logger.info(f"📖 GameHistoryRetrieval artifact: {artifact_preview}")

            history_context, game_id = self._resolve_history_context(query, last_artifact)

            llm_structured = self._llm.with_structured_output(ToolOutput)
            response: ToolOutput = (get_game_history_retrieval_prompt_template() | llm_structured).invoke({
                "query": query,
                "context": history_context,
            })  # type: ignore
            logger.info(f"GameHistoryRetrieval response: {response}")
            return response.answer, game_id

        except Exception as e:
            error_msg = f"Error in game_history_retrieval: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if run_tree:
                run_tree.end(error=error_msg)
            return f"An error occurred while retrieving game history. Details: {str(e)}. Please try again or stop the execution.", None
