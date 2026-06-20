"""Standalone Qdrant access service.

Owns the Qdrant client lifecycle (both sync and async) so call sites never
construct their own client — they just pass a collection name + query vector and
optional search params (limit, score_threshold, query_filter, with_vectors,
exact). Clients are created lazily on first use so the async client binds to the
running event loop rather than import time.
"""

import logging
from typing import Any, List, Optional

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Filter, SearchParams

from app.config.settings import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """Shared Qdrant client wrapper exposing sync (`search`) and async (`asearch`) lookups."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = True,
        timeout: int = 20,
    ) -> None:
        self._url = url or settings.QDRANT_URL
        self._api_key = api_key or settings.QDRANT_API_KEY
        self._prefer_grpc = prefer_grpc
        self._timeout = timeout
        self._sync_client: Optional[QdrantClient] = None
        self._async_client: Optional[AsyncQdrantClient] = None

    # ------------------------------------------------------------------
    # Lazy clients
    # ------------------------------------------------------------------
    @property
    def sync_client(self) -> QdrantClient:
        if self._sync_client is None:
            self._sync_client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
                prefer_grpc=self._prefer_grpc,
                check_compatibility=False,
                timeout=self._timeout,
            )
        return self._sync_client

    @property
    def async_client(self) -> AsyncQdrantClient:
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(
                url=self._url,
                api_key=self._api_key,
                prefer_grpc=self._prefer_grpc,
                check_compatibility=False,
                timeout=self._timeout,
            )
        return self._async_client

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        collection_name: str,
        vector: List[float],
        *,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        exact: bool = False,
    ) -> List[Any]:
        """Synchronous vector search. Returns the list of scored points."""
        result = self.sync_client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
            search_params=SearchParams(exact=exact) if exact else None,
        )
        return result.points

    async def asearch(
        self,
        collection_name: str,
        vector: List[float],
        *,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        exact: bool = False,
    ) -> List[Any]:
        """Asynchronous vector search. Returns the list of scored points."""
        result = await self.async_client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
            search_params=SearchParams(exact=exact) if exact else None,
        )
        return result.points

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """Establish the sync gRPC connection so the first real search has no cold start."""
        try:
            collections = self.sync_client.get_collections()
            logger.info(f"✅ Qdrant sync client warmed up ({len(collections.collections)} collections)")
        except Exception as e:
            logger.warning(f"⚠️ Qdrant sync warmup failed (non-fatal): {e}")

    async def awarmup(self) -> None:
        """Establish the async gRPC connection (binds to the running event loop)."""
        try:
            await self.async_client.get_collections()
            logger.info("✅ Qdrant async client warmed up")
        except Exception as e:
            logger.warning(f"⚠️ Qdrant async warmup failed (non-fatal): {e}")

    def close(self) -> None:
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None


# Module-level singleton — import this everywhere a Qdrant search is needed.
qdrant_service = QdrantService()
