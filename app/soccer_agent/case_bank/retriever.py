import asyncio
import json
import logging
from collections import Counter
from typing import List, Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config.settings import settings
from app.services.qdrant_service import qdrant_service

from app.cache.standard_cache import standard_cache

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.QDRANT_CASE_BANK_COLLECTION_NAME
TOP_POSITIVE = 3
TOP_NEGATIVE = 6


class CaseBankRetriever:
    """
    Retrieves few-shot planning examples from Qdrant planning_case_bank collection.

    Flow:
      1. Two parallel searches: top-3 positive + top-6 negative (filtered by has_media)
      2. Group all 9 results by use_case → pick top-3 use_cases by hit count
      3. Format into a structured string for injection into the planning prompt
    """

    def __init__(self):
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            output_dimensionality=768,
            google_api_key=settings.GOOGLE_API_KEY,
            task_type="RETRIEVAL_QUERY",
        )

    @standard_cache.cache(ttl=60*60*24) # Cache embeddings for 24h
    async def _embed(self, text: str) -> List[float]:
        return await self._embeddings.aembed_query(text)

    async def embed_query(self, text: str) -> List[float]:
        """Public RETRIEVAL_QUERY embedding (24h cached). Shared by context_retrieval
        so the same query vector feeds the cache, the retriever, and long-term memory."""
        return await self._embed(text)

    async def warmup(self) -> None:
        """Pre-warm the embedding model and exercise the real search path at startup."""
        try:
            dummy_vec = await self._embed("warmup")
            await qdrant_service.asearch(
                collection_name=COLLECTION_NAME,
                vector=dummy_vec[:768],
                limit=1,
                with_payload=False,
                exact=True,
            )
            logger.info("✅ CaseBankRetriever warmed up (gRPC + embedding ready)")
        except Exception as e:
            logger.warning(f"⚠️ CaseBankRetriever warmup failed (non-fatal): {e}")

    async def _search(
        self, vector: List[float], has_media: bool, label: str, top_k: int
    ) -> list:
        try:
            return await qdrant_service.asearch(
                collection_name=COLLECTION_NAME,
                vector=vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="label", match=MatchValue(value=label)),
                    ]
                ),
                limit=top_k,
                with_payload=True,
                exact=True,
            )
        except Exception as e:
            logger.error(f"CaseBankRetriever search error (label={label}): {e}", exc_info=True)
            return []

    def _format_examples(self, grouped: dict) -> str:
        if not grouped:
            return ""
        parts = ["## Retrieved Examples"]
        for use_case, examples in grouped.items():
            parts.append(f"\n### Use Case: {use_case}")
            for ex in examples:
                p = ex.payload
                label = p.get("label", "unknown")
                tool_chains = p.get("tool_chains", "[]")
                sub_queries = p.get("sub_queries", "[]")
                reasoning = p.get("reasoning", "")
                try:
                    tc_str = json.dumps(json.loads(tool_chains), separators=(",", ":"))
                    sq_str = json.dumps(json.loads(sub_queries), separators=(",", ":"))
                except Exception:
                    tc_str = tool_chains
                    sq_str = sub_queries
                need_call_tools = p.get("need_call_tools", True)
                tag = "✅ Correct" if label == "positive" else "❌ Wrong"
                parts.append(
                    f"- **[{tag}]** need_call_tools={need_call_tools} | tool_chains={tc_str} | sub_queries={sq_str}\n"
                    f"  Reasoning: {reasoning}"
                )
        return "\n".join(parts)

    async def retrieve(
        self, query: str, has_media: bool, precomputed_embedding: Optional[List[float]] = None
    ) -> str:
        vector = precomputed_embedding if precomputed_embedding is not None else await self._embed(query)

        positive_hits, negative_hits = await asyncio.gather(
            self._search(vector, has_media, "positive", TOP_POSITIVE),
            self._search(vector, has_media, "negative", TOP_NEGATIVE),
        )

        all_hits = positive_hits + negative_hits
        if not all_hits:
            logger.info("CaseBankRetriever: no results found")
            return ""

        # Count use_case occurrences to select top-3 representative use_cases
        use_case_counter: Counter = Counter()
        for hit in all_hits:
            uc = hit.payload.get("use_case", "unknown")
            use_case_counter[uc] += 1

        top_use_cases = [uc for uc, _ in use_case_counter.most_common(3)]

        # Group hits by use_case (preserving positive-first order within each group)
        grouped: dict[str, list] = {uc: [] for uc in top_use_cases}
        for hit in positive_hits + negative_hits:
            uc = hit.payload.get("use_case", "unknown")
            if uc in grouped:
                grouped[uc].append(hit)

        formatted = self._format_examples(grouped)
        logger.info(
            f"CaseBankRetriever: retrieved examples for use_cases={top_use_cases}"
        )
        return formatted
