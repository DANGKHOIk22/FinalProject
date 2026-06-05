import logging
from typing import Annotated, Optional, Type

from dns import resolver
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.tools import InjectedState, BaseTool
from pydantic import BaseModel, Field
from pymongo import ASCENDING, MongoClient

from app.config.settings import settings

logger = logging.getLogger(__name__)

_mongo_client: Optional[MongoClient] = None


def _get_collection():
    global _mongo_client
    if _mongo_client is None:
        _r = resolver.Resolver(configure=False)
        _r.nameservers = ["8.8.8.8", "1.1.1.1"]
        resolver.default_resolver = _r
        _mongo_client = MongoClient(settings.MONGO_SRV)
    return _mongo_client[settings.SOCCER_DB_NAME][settings.TRANSCRIPTION_COLLECTION_NAME]


class CommentaryExtractionInput(BaseModel):
    start_time: float = Field(
        description="Start of time range in seconds."
    )
    end_time: float = Field(
        description="End of time range in seconds. "
    )
    execution_agent_state: Annotated[dict, InjectedState] = Field(
        description="Injected worker state — provides video_id and video_current_time."
    )


class CommentaryExtractionTool(BaseTool):
    name: str = "commentary_extraction"
    description: str = (
        "Extracts transcript text from a soccer video stored in MongoDB. "
        "Provide start_time and end_time in seconds — use video_current_time from agent state as reference "
        "(e.g. current_time - 5 to current_time for recent commentary). "
        "Returns the concatenated transcript text for the requested time window."
    )
    args_schema: Type[BaseModel] = CommentaryExtractionInput  # type: ignore

    def _run(
        self,
        start_time: float,
        end_time: float,
        execution_agent_state: Annotated[dict, InjectedState],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        video_id = execution_agent_state.get("video_id")
        if not video_id:
            return "No active HLS session (video_id not found in state). This tool requires a video."

        try:
            ts_min, ts_max = start_time, end_time

            collection = _get_collection()
            docs = list(
                collection.find(
                    {"video_id": video_id, "timestamp": {"$gte": ts_min, "$lte": ts_max}},
                    {"transcript": 1, "_id": 0},
                    sort=[("timestamp", ASCENDING)],
                )
            )

            if not docs:
                return (
                    f"No transcript found for video_id='{video_id}' "
                    f"in range ({ts_min:.1f}s – {ts_max:.1f}s)."
                )

            result = "\n".join(d["transcript"] for d in docs if d.get("transcript"))
            logger.info(
                f"✅ commentary_extraction: {len(docs)} chunks | "
                f"video_id={video_id} | {ts_min:.1f}s–{ts_max:.1f}s"
            )
            return result

        except Exception as e:
            error_msg = f"Error in commentary_extraction: {e}"
            logger.error(error_msg)
            return error_msg
