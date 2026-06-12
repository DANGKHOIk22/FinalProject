import hashlib
import logging
from typing import List, Optional
from redisvl.query.filter import Tag
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config.settings import settings

logger = logging.getLogger(__name__)


def _material_tag(material: Optional[List[str]]) -> str:
    """Build a stable tag value for a material list.

    Hashed because RediSearch splits tag values on commas (media URLs often
    contain them) and the tag must not depend on list order.
    """
    if not material:
        return "None"
    return hashlib.md5("|".join(sorted(material)).encode()).hexdigest()


class SubQuerySemanticCache:
    def __init__(self, threshold: float = 0.05, ttl: int = 60 * 60):
        self.threshold = threshold
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                output_dimensionality=768,
                api_key=settings.GOOGLE_API_KEY,
                task_type="RETRIEVAL_QUERY"
            )
            config = RedisConfig(
                index_name="semantic_cache",
                redis_url=settings.REDIS_URL,
                distance_metric="COSINE",
                embedding_dimensions=768,
                metadata_schema=[
                    {"name": "response", "type": "text"},
                    {"name": "material", "type": "tag"}
                ]
            )
            self.vector_store = RedisVectorStore(
                config = config,
                embeddings = self.embeddings,
                ttl = ttl
            )
            self.is_active = True
            logger.info("✅ Redis Vector Store Cache initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RedisVectorStore: {e}")
            self.is_active = False

    def check(self, query: str, material: Optional[List[str]] = None) -> str | None:
        if not self.is_active or not query:
            return None
            
        # Format the material list to a stable tag to match on filter
        material_str = _material_tag(material)
        logger.debug(f"Checking Semantic cache for sub-query: '{query}' with material: '{material_str}'")
        
        try:
            filter_condition = Tag("material") == material_str
            docs = self.vector_store.similarity_search_with_score(
                query=query,
                k=1,
                filter=filter_condition,
                distance_threshold = self.threshold
            )
            if docs:
                logger.info(f"🎯 Semantic cache HIT for query: '{query}'")
                return docs[0][0].metadata.get("response")
                    
            logger.debug(f"Semantic cache MISS for query: '{query}'")
        except Exception as e:
            logger.error(f"Semantic cache lookup error: {e}")
            
        return None

    def set(self, query: str, response: str, material: Optional[List[str]] = None):
        if not self.is_active or not query or not response:
            return

        material_str = _material_tag(material)
        try:
            metadata = {
                "response": response,
                "material": material_str
            }
            self.vector_store.add_texts(
                texts=[query],
                metadatas=[metadata]
            )
            logger.debug(f"Saved worker result to Semantic cache for query: '{query}' with material: '{material_str}'")
        except Exception as e:
            logger.error(f"Semantic cache update error: {e}")

# Global instance with proximity threshold
semantic_cache = SubQuerySemanticCache(threshold=0.15, ttl=60 * 60)
