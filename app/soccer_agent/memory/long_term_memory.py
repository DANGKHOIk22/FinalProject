import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict
import uuid
import numpy as np

import psycopg
from psycopg.rows import dict_row
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.soccer_agent.memory.chat_history import get_connection_pool
from app.cache.standard_cache import standard_cache

logger = logging.getLogger(__name__)

class LongTermMemoryManager:
    """
    Manages Long-term Memory using PostgreSQL with pgvector.
    Stores and retrieves tool artifacts (Wiki/News) as vector embeddings.
    """
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200,
        )
        self.pool = get_connection_pool()

    @standard_cache.cache(ttl=60*60) # Cache for 1 hour only to save Redis space
    async def _get_embedding(self, text: str) -> List[float]:
        return await self.embeddings.aembed_query(text)

    async def upsert_memory_smart(
        self, 
        user_id: str, 
        entity_name: str, 
        content: str, 
        metadata: Optional[Dict] = None,
        similarity_threshold: float = 0.95
    ) -> None:
        """
        Smartly store content in pgvector by checking for existing similar chunks first.
        Prevents redundant data entry.
        """
        if not content or len(content.strip()) < 50:
            return

        try:
            entity_name_norm = entity_name.lower().strip()
            chunks = self.text_splitter.split_text(content)
            
            # 1. Get existing embeddings for this entity to compare
            existing_embeddings = []
            def get_existing():
                with self.pool.getconn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT embedding FROM user_long_term_memory WHERE user_id = %s AND entity_name = %s",
                            (user_id, entity_name_norm)
                        )
                        return [np.array(row[0]) for row in cur.fetchall()]
            
            existing_vectors = await asyncio.to_thread(get_existing)
            
            # 2. Process each new chunk
            new_records = []
            for chunk in chunks:
                new_vec = await self._get_embedding(chunk)
                new_vec_np = np.array(new_vec)
                
                # Check similarity against ALL existing vectors for this entity
                is_duplicate = False
                for old_vec in existing_vectors:
                    # Cosine Similarity = (A . B) / (||A|| * ||B||)
                    # Since embeddings are usually normalized, just dot product
                    similarity = np.dot(new_vec_np, old_vec) / (np.linalg.norm(new_vec_np) * np.linalg.norm(old_vec))
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    new_records.append((chunk, new_vec))
                    # Add to existing_vectors list for this session to avoid internal duplication
                    existing_vectors.append(new_vec_np)

            if not new_records:
                logger.info(f"[LongTermMemory] No new unique chunks for '{entity_name_norm}'")
                return

            # 3. Batch insert new unique chunks
            def sync_insert():
                with self.pool.getconn() as conn:
                    with conn.cursor() as cur:
                        for chunk, vec in new_records:
                            cur.execute(
                                """
                                INSERT INTO user_long_term_memory 
                                (user_id, entity_name, content, embedding, metadata)
                                VALUES (%s, %s, %s, %s::vector, %s)
                                """,
                                (user_id, entity_name_norm, chunk, vec, json.dumps(metadata) if metadata else None)
                            )
                        conn.commit()

            await asyncio.to_thread(sync_insert)
            logger.info(f"[LongTermMemory] Added {len(new_records)} unique chunks for '{entity_name_norm}'")

        except Exception as e:
            logger.error(f"[LongTermMemory] Smart upsert failed: {e}", exc_info=True)

    async def retrieve_memory(
        self, 
        user_id: str, 
        query: str, 
        top_k: int = 5, 
        precomputed_embedding: Optional[List[float]] = None
    ) -> List[Dict]:
        try:
            query_embedding = precomputed_embedding if precomputed_embedding else await self._get_embedding(query)
            
            def sync_retrieve():
                with self.pool.getconn() as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute(
                            """
                            SELECT content, entity_name, metadata, created_at,
                                   (1 - (embedding <=> %s::vector)) as similarity
                            FROM user_long_term_memory
                            WHERE user_id = %s
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                            """,
                            (query_embedding, user_id, query_embedding, top_k)
                        )
                        return cur.fetchall()

            return await asyncio.to_thread(sync_retrieve)
        except Exception as e:
            logger.error(f"[LongTermMemory] Retrieval failed: {e}", exc_info=True)
            return []

    def _cleanup_old_news_sync(self):
        try:
            with self.pool.getconn() as conn:
                with conn.cursor() as cur:
                    threshold = datetime.now() - timedelta(hours=24)
                    cur.execute("DELETE FROM user_long_term_memory WHERE created_at < %s", (threshold,))
                    conn.commit()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

long_term_memory_manager = LongTermMemoryManager()
