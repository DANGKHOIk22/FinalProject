import hashlib
import logging
from typing import List, Optional
from redisvl.query.filter import Tag, Num
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config.settings import settings

logger = logging.getLogger(__name__)

_NO_VIDEO = "__none__"
_VIDEO_TIME_WINDOW = 5.0  # seconds — cache hit valid only within this window before current_time


def _image_tag(image_id: Optional[List[str]]) -> str:
    """Build a stable tag value for an image_id list.

    Hashed because RediSearch splits tag values on commas and the tag
    must not depend on list order.
    """
    if not image_id:
        return "None"
    return hashlib.md5("|".join(sorted(image_id)).encode()).hexdigest()


class SemanticCache:
    def __init__(self, index_name: str, threshold: float = 0.05, ttl: int = 60 * 60):
        self.threshold = threshold
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                output_dimensionality=768,
                api_key=settings.GOOGLE_API_KEY,
                task_type="RETRIEVAL_QUERY"
            )
            config = RedisConfig(
                index_name=index_name,
                redis_url=settings.REDIS_URL,
                distance_metric="COSINE",
                embedding_dimensions=768,
                metadata_schema=[
                    {"name": "response", "type": "text"},
                    {"name": "image_id", "type": "tag"},
                    {"name": "video_id", "type": "tag"},
                    {"name": "game_id", "type": "tag"},
                    {"name": "timestamp", "type": "numeric"},
                ]
            )
            self.vector_store = RedisVectorStore(
                config=config,
                embeddings=self.embeddings,
                ttl=ttl
            )
            self.is_active = True
            logger.info("✅ Redis Vector Store Cache initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RedisVectorStore: {e}")
            self.is_active = False

    def check(
        self,
        query: str,
        additional_material: Optional[dict] = None,
        current_time: Optional[float] = None,
    ) -> str | None:
        if not self.is_active or not query:
            return None

        additional_material = additional_material or {}
        game_id = additional_material.get("game_id")
        image_id = additional_material.get("image_id")
        video_id = additional_material.get("video_id")

        try:
            if game_id is not None:
                # Video path: filter by game_id + timestamp within [current_time - 5s, current_time]
                ts = current_time if current_time is not None else 0.0
                ts_min = max(0.0, ts - _VIDEO_TIME_WINDOW)
                filter_condition = (
                    (Tag("game_id") == game_id)
                    & (Num("timestamp") >= ts_min)
                    & (Num("timestamp") <= ts)
                )
                logger.debug(
                    f"Checking Semantic cache | game_id='{game_id}' "
                    f"window=[{ts_min:.1f}s, {ts:.1f}s] query='{query}'"
                )
            else:
                # Non-video path: filter by image_id & video_id, exclude video-scoped entries
                image_id_str = _image_tag(image_id)
                video_id_str = str(video_id) if video_id else "None"
                filter_condition = (
                    (Tag("image_id") == image_id_str)
                    & (Tag("video_id") == video_id_str)
                    & (Tag("game_id") == _NO_VIDEO)
                )
                logger.debug(
                    f"Checking Semantic cache | image_id='{image_id_str}' "
                    f"video_id='{video_id_str}' query='{query}'"
                )

            docs = self.vector_store.similarity_search_with_score(
                query=query,
                k=1,
                filter=filter_condition,
                distance_threshold=self.threshold
            )
            if docs:
                logger.info(f"🎯 Semantic cache HIT for query: '{query}'")
                return docs[0][0].metadata.get("response")

            logger.info(f"⏳ Semantic cache MISS for query: '{query}'")
        except Exception as e:
            logger.error(f"Semantic cache lookup error: {e}")

        return None

    def set(
        self,
        query: str,
        response: str,
        additional_material: Optional[dict] = None,
        current_time: Optional[float] = None,
    ):
        if not self.is_active or not query or not response:
            return

        additional_material = additional_material or {}
        game_id = additional_material.get("game_id")
        image_id = additional_material.get("image_id")
        video_id = additional_material.get("video_id")

        try:
            if game_id is not None:
                metadata = {
                    "response": response,
                    "image_id": "None",
                    "video_id": "None",
                    "game_id": game_id,
                    "timestamp": current_time if current_time is not None else 0.0,
                }
                logger.debug(
                    f"Saved to Semantic cache | game_id='{game_id}' "
                    f"timestamp={metadata['timestamp']} query='{query}'"
                )
            else:
                image_id_str = _image_tag(image_id)
                video_id_str = str(video_id) if video_id else "None"
                metadata = {
                    "response": response,
                    "image_id": image_id_str,
                    "video_id": video_id_str,
                    "game_id": _NO_VIDEO,
                    "timestamp": 0.0,
                }
                logger.debug(
                    f"Saved to Semantic cache | image_id='{image_id_str}' "
                    f"video_id='{video_id_str}' query='{query}'"
                )

            self.vector_store.add_texts(texts=[query], metadatas=[metadata])
        except Exception as e:
            logger.error(f"Semantic cache update error: {e}")


# Worker answers only. Context retrieval caches few-shot cases via CaseBankCache
# (per query + has_media) and never caches per-user long-term memory.
# After deploying, drop the old shared index: FT.DROPINDEX semantic_cache DD
sub_query_cache = SemanticCache(index_name="sub_query_cache", threshold=0.1, ttl=3600)
