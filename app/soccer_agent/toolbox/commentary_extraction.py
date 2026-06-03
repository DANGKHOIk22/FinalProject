import logging
from typing import Literal, Optional, Type

from dns import resolver
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator
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
    intent: Literal["current", "recent", "specific"] = Field(
        description="'current' = last 5s from current_time, 'recent' = last 15s, 'specific' = explicit start_time/end_time range"
    )
    video_id: str = Field(description="HLS video session ID")
    current_time: Optional[float] = Field(
        default=None,
        description="Current playback position in seconds. Required for intent='current' or 'recent'.",
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Start of time range in seconds. Required for intent='specific'.",
    )
    end_time: Optional[float] = Field(
        default=None,
        description="End of time range in seconds. Required for intent='specific'.",
    )

    @model_validator(mode="after")
    def check_required_fields(self):
        if self.intent in ("current", "recent") and self.current_time is None:
            raise ValueError(f"'current_time' is required for intent='{self.intent}'")
        if self.intent == "specific":
            if self.start_time is None or self.end_time is None:
                raise ValueError("'start_time' and 'end_time' are required for intent='specific'")
            if self.start_time > self.end_time:
                raise ValueError("'start_time' must be <= 'end_time'")
        return self


class CommentaryExtractionTool(BaseTool):
    name: str = "commentary_extraction"
    description: str = (
        "Extracts transcript text from a soccer video stored in MongoDB. "
        "Use intent='current' for the last 5 seconds of commentary, 'recent' for the last 15 seconds, "
        "or 'specific' with start_time and end_time (in seconds) for an explicit range. "
        "Returns the concatenated transcript text for the requested time window."
    )
    args_schema: Type[BaseModel] = CommentaryExtractionInput  # type: ignore

    def _run(
        self,
        intent: Literal["current", "recent", "specific"],
        video_id: str,
        current_time: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            if intent == "current":
                ts_min, ts_max = current_time - 5.0, current_time  # type: ignore[operator]
            elif intent == "recent":
                ts_min, ts_max = current_time - 15.0, current_time  # type: ignore[operator]
            else:
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
