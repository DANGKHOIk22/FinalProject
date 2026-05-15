import logging
from typing import List, Optional
from redisvl.query.filter import Tag
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config.settings import settings

logger = logging.getLogger(__name__)

class SubQuerySemanticCache:
    def __init__(self, threshold: float = 0.1, ttl: int = 60 * 60):
        self.threshold = threshold
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                output_dimensionality=768,
                google_api_key=settings.GOOGLE_API_KEY,
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
            
        # Format the material list to string to match on filter
        material_str = ", ".join(material) if material else "None"
        logger.info(f"🔍 Checking Semantic cache for sub-query: '{query}' with material: '{material_str}'")
        
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
                    
            logger.info(f"⏳ Semantic cache MISS for query: '{query}'")
        except Exception as e:
            logger.error(f"Semantic cache lookup error: {e}")
            
        return None

    def cache(self, ttl: Optional[int] = None):
        """
        Decorator for semantic caching.
        Works for async functions.
        """
        import functools
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Try to find a 'query' or 'user_query' in args/kwargs
                query = kwargs.get("user_query") or kwargs.get("query")
                if not query and args:
                    # Heuristic: first string arg is likely the query
                    for arg in args:
                        if isinstance(arg, str):
                            query = arg
                            break
                
                if not query:
                    return await func(*args, **kwargs)

                # Check cache
                cached_res = self.check(query)
                if cached_res:
                    # If the function returns a dict, try to parse JSON
                    if cached_res.startswith("{") and cached_res.endswith("}"):
                        try:
                            import json
                            return json.loads(cached_res)
                        except:
                            pass
                    return cached_res

                # Execute
                result = await func(*args, **kwargs)

                # Store (serialize if dict)
                res_to_store = result
                if isinstance(result, dict):
                    import json
                    res_to_store = json.dumps(result)
                
                if res_to_store:
                    self.set(query, res_to_store)
                
                return result
            return wrapper
        return decorator

    def set(self, query: str, response: str, material: Optional[List[str]] = None):
        if not self.is_active or not query or not response:
            return
            
        material_str = ", ".join(material) if material else "None"
        try:
            metadata = {
                "response": response,
                "material": material_str
            }
            self.vector_store.add_texts(
                texts=[query],
                metadatas=[metadata]
            )
            logger.info(f"💾 Saved worker result to Semantic cache for query: '{query}' with material: '{material_str}'")
        except Exception as e:
            logger.error(f"Semantic cache update error: {e}")

# Global instance with proximity threshold
semantic_cache = SubQuerySemanticCache(threshold=0.15, ttl=60 * 60)
