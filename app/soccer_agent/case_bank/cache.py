import logging
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from redisvl.query.filter import Tag

from app.config.settings import settings

logger = logging.getLogger(__name__)


class CaseBankCache:
    """
    Redis-backed semantic cache for case_bank retrieval results.
    Key: embedding of clarified_query + has_media flag.
    Value: pre-formatted 9-examples string ready to inject into planning prompt.
    """

    def __init__(self, threshold: float = 0.15, ttl: int = 60 * 60):
        self.threshold = threshold
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                output_dimensionality=768,
                google_api_key=settings.GOOGLE_API_KEY,
                task_type="RETRIEVAL_QUERY",
            )
            config = RedisConfig(
                index_name="case_bank_cache",
                redis_url=settings.REDIS_URL,
                distance_metric="COSINE",
                embedding_dimensions=768,
                metadata_schema=[
                    {"name": "examples", "type": "text"},
                    {"name": "has_media", "type": "tag"},
                ],
            )
            self.vector_store = RedisVectorStore(
                config=config,
                embeddings=self.embeddings,
                ttl=ttl,
            )
            self.is_active = True
            logger.info("✅ CaseBankCache initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize CaseBankCache: {e}")
            self.is_active = False

    def get(
        self, query: str, has_media: bool, precomputed_embedding: Optional[List[float]] = None
    ) -> Optional[str]:
        if not self.is_active or not query:
            return None
        media_tag = "true" if has_media else "false"
        logger.info(f"🔍 CaseBankCache check: '{query[:80]}' has_media={has_media}")
        try:
            filter_condition = Tag("has_media") == media_tag
            # Reuse the shared query embedding when provided — avoids re-embedding.
            if precomputed_embedding is not None:
                docs = self.vector_store.similarity_search_with_score_by_vector(
                    embedding=precomputed_embedding,
                    k=1,
                    filter=filter_condition,
                    distance_threshold=self.threshold,
                )
            else:
                docs = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=1,
                    filter=filter_condition,
                    distance_threshold=self.threshold,
                )
            if docs:
                logger.info("🎯 CaseBankCache HIT")
                return docs[0][0].metadata.get("examples")
            logger.info("⏳ CaseBankCache MISS")
        except Exception as e:
            logger.error(f"CaseBankCache lookup error: {e}")
        return None

    def set(self, query: str, has_media: bool, examples: str) -> None:
        if not self.is_active or not query or not examples:
            return
        media_tag = "true" if has_media else "false"
        try:
            self.vector_store.add_texts(
                texts=[query],
                metadatas=[{"examples": examples, "has_media": media_tag}],
            )
            logger.info(f"💾 CaseBankCache saved for query: '{query[:80]}'")
        except Exception as e:
            logger.error(f"CaseBankCache save error: {e}")


case_bank_cache = CaseBankCache()
