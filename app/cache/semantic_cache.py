import logging
from langchain_redis import RedisSemanticCache
from langchain_core.outputs import Generation
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config.settings import settings

logger = logging.getLogger(__name__)

class SubQuerySemanticCache:
    def __init__(self, threshold: float = 0.1, ttl: int = 60 * 60 ):
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=settings.GOOGLE_API_KEY,
            )
            self.cache = RedisSemanticCache(
                redis_url=settings.REDIS_URL,
                embeddings=self.embeddings,
                distance_threshold=threshold,
                ttl=ttl,
                name="llm_cache",
                prefix="llmcache",
            )
            self.llm_string = "sub_query_worker"
            self.is_active = True
            logger.info("✅ Redis Semantic Cache initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RedisSemanticCache: {e}")
            self.is_active = False

    def check(self, query: str) -> str | None:
        logger.info(f"🔍 Checking semantic cache for sub-query: {query}")
        if not self.is_active or not query:
            return None
        try:
            result = self.cache.lookup(query, self.llm_string)
            if result and len(result) > 0:
                logger.info(f"🎯 Semantic cache HIT for query: '{query}'")
                return result[0].text
            logger.info(f"⏳ Semantic cache MISS for query: '{query}'")
        except Exception as e:
            logger.error(f"Semantic cache lookup error: {e}")
        return None

    def set(self, query: str, response: str):
        if not self.is_active or not query or not response:
            return
        try:
            self.cache.update(query, self.llm_string, [Generation(text=response)])
            logger.info(f"💾 Saved worker result to semantic cache for query: '{query}'")
        except Exception as e:
            logger.error(f"Semantic cache update error: {e}")

# Global instance with similarity threshold > 0.9
semantic_cache = SubQuerySemanticCache(threshold=0.1, ttl=60 * 60 )
