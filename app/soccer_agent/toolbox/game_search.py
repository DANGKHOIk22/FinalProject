import logging
from typing import Type, Optional, Literal, Tuple, Any

from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.output_parsers import PydanticOutputParser
from pymongo import MongoClient
from dns import resolver

from app.config.settings import settings
from app.schema.match import MatchInfo
from app.soccer_agent.factory.llm_provider import get_llm
from app.soccer_agent.prompts.toolbox.game_search import (
    get_extraction_prompt_template,
    get_match_selection_prompt_template,
)

logger = logging.getLogger(__name__)


class GameSearchInput(BaseModel):
    query: str = Field(description="Câu truy vấn tự nhiên về trận đấu bóng đá cần tìm kiếm.")


class FinalResult(BaseModel):
    game_id: Optional[str] = Field(
        description="game_id của trận đấu nếu tìm thấy chính xác. Nếu không, để null.",
        default=None,
    )
    response_llm: str = Field(description="Lời giải thích chi tiết về việc tìm thấy hay không.")


class GameSearchTool(BaseTool):
    name: str = "game_search"
    description: str = """
    Ability: Given certain information regarding a match, this tool retrieves the corresponding game from the soccer match database.
    The games pertain to six major European leagues (England Premier, Germany Bundesliga, Italy Serie A, Spain La Liga, France Ligue 1,
    and the European Champions League) spanning the years 2017-2024.
    Query Input: Simply the original inquiry as the query input here.
    Output: Returns a summary message and the game_id of the identified game as an artifact.
    Remark: This tool must be utilized initially to acquire the game's context.
    """
    args_schema: Type[BaseModel] = GameSearchInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    _llm: Any = PrivateAttr()
    _parser: PydanticOutputParser = PrivateAttr()
    _collection: Any = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._llm = get_llm("tool")
        self._parser = PydanticOutputParser(pydantic_object=MatchInfo)

        _resolver = resolver.Resolver(configure=False)
        _resolver.nameservers = ['8.8.8.8', '1.1.1.1']
        resolver.default_resolver = _resolver

        mongo_client = MongoClient(settings.MONGO_SRV)
        self._collection = mongo_client[settings.SOCCER_DB_NAME][settings.GAME_COLLECTION_NAME]

    def _extract_match_info(self, query: str) -> MatchInfo:
        prompt = get_extraction_prompt_template()
        chain = prompt | self._llm | self._parser
        return chain.invoke({
            "question": query,
            "format_instructions": self._parser.get_format_instructions(),
        })

    def _build_mongo_filter(self, info: MatchInfo) -> dict:
        f: dict = {}
        if info.league != "unknown":
            f["league"] = info.league
        if info.season != "unknown":
            f["season"] = info.season
        if info.year != "unknown":
            f.setdefault("date", {})
            f["date"]["$regex"] = f"^{info.year}"
        if info.month != "unknown":
            m = info.month.lstrip("0").zfill(2)
            f.setdefault("date", {})
            f["date"]["$regex"] = f"^\\d{{4}}-{m}"

        t1 = info.team1 if info.team1 != "unknown" else ""
        t2 = info.team2 if info.team2 != "unknown" else ""

        if t1 and t2:
            f["$or"] = [
                {"home_team": {"$regex": t1, "$options": "i"}, "away_team": {"$regex": t2, "$options": "i"}},
                {"home_team": {"$regex": t2, "$options": "i"}, "away_team": {"$regex": t1, "$options": "i"}},
            ]
        elif t1:
            f["$or"] = [
                {"home_team": {"$regex": t1, "$options": "i"}},
                {"away_team": {"$regex": t1, "$options": "i"}},
            ]
        elif t2:
            f["$or"] = [
                {"home_team": {"$regex": t2, "$options": "i"}},
                {"away_team": {"$regex": t2, "$options": "i"}},
            ]
        return f

    def _retrieve_candidates(self, info: MatchInfo) -> list[dict]:
        mongo_filter = self._build_mongo_filter(info)
        projection = {
            "game_id": 1, "league": 1, "season": 1,
            "date": 1, "home_team": 1, "away_team": 1, "score": 1, "_id": 0,
        }
        return list(self._collection.find(mongo_filter, projection).limit(20))

    def _finalize_candidate_selection(
        self, candidates: list[dict], info: MatchInfo, question: str
    ) -> Tuple[str, Optional[str]]:
        if not candidates:
            return (
                "We did not find the match you mentioned in the database. "
                "Stop the execution and ask user give more specific information.",
                None,
            )

        if len(candidates) == 1:
            g = candidates[0]
            return f"Found match: {g['home_team']} vs {g['away_team']} ({g['date']}).", g["game_id"]

        candidate_text = ""
        for i, g in enumerate(candidates, 1):
            candidate_text += (
                f"\nCandidate {i}:\n"
                f"  - Date: {g['date']}, League: {g['league']}\n"
                f"  - Match: {g['home_team']} vs {g['away_team']}\n"
                f"  - Score: {g['score']}, game_id: {g['game_id']}\n"
            )

        prompt = get_match_selection_prompt_template()
        structured_llm = self._llm.with_structured_output(FinalResult)
        response: FinalResult = (prompt | structured_llm).invoke({
            "question": question,
            "info": info.model_dump_json(),
            "candidates": candidate_text,
        })  # type: ignore
        return response.response_llm, response.game_id

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Optional[str]]:
        logger.info(f"🔎 Game Search Query: {query}")
        try:
            info = self._extract_match_info(query)
            logger.debug(f"Extracted Info: {info}")
            candidates = self._retrieve_candidates(info)
            content, artifact = self._finalize_candidate_selection(candidates, info, query)
            logger.info(f"✅ Search Result: {content} | game_id: {artifact}")
            return content, artifact
        except Exception as e:
            logger.error(f"Error in Game Search: {e}", exc_info=True)
            return (
                f"An error occurred while searching for the game. Details: {str(e)}. "
                "Please check your query or try again.",
                None,
            )
