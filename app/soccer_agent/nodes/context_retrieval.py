import logging
import asyncio
from datetime import datetime
import uuid
from langgraph.graph.state import RunnableConfig
from langchain.messages import HumanMessage
from app.config.settings import settings
from app.schema.soccer_agent.state import AgentState
from app.soccer_agent.memory.long_term_memory import long_term_memory_manager
from app.soccer_agent.case_bank.retriever import CaseBankRetriever
from app.soccer_agent.case_bank.cache import case_bank_cache

logger = logging.getLogger(__name__)

class ContextRetrievalNode:
    """
    Unified Node for retrieving both Case Bank (Few-shot) and 
    Long-term Memory (Knowledge) context.
    """
    def __init__(self, case_bank_retriever: CaseBankRetriever):
        self.case_bank_retriever = case_bank_retriever

    async def retrieve_context_node(self, state: AgentState, config: RunnableConfig) -> dict:
        # 1. Identify User Query
        messages = state.get("messages", [])
        user_query = str(messages[-1].text) if isinstance(messages[-1], HumanMessage) else None
        metadata = config.get("metadata", {})
        thread_id = metadata.get("thread_id", str(uuid.uuid4()))
        user_id = str(config.get("configurable", {}).get("user_id"))
        has_media = bool(state.get("additional_material"))
        
        if not user_query:
            return {"long_term_context": "", "retrieved_cases": ""}

        # 2. Get Embedding ONCE
        try:
            query_embedding = await long_term_memory_manager._get_embedding(user_query)
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}", exc_info=True)
            query_embedding = None

        # 3. Parallel Retrieval
        # Only the case-bank examples are cached (global knowledge, keyed by
        # query + has_media). Long-term memory is per-user and must NOT be
        # cached without user scoping — see cache-layer fix 2026-06-12.
        async def fetch_cases():
            try:
                cached_cases = await asyncio.to_thread(
                    case_bank_cache.get, user_query, has_media
                )
                if cached_cases:
                    return cached_cases

                cases = await self.case_bank_retriever.retrieve(user_query, has_media)
                if cases:
                    await asyncio.to_thread(
                        case_bank_cache.set, user_query, has_media, cases
                    )
                return cases
            except Exception as e:
                logger.error(f"Error fetching cases: {e}")
                return ""

        async def fetch_long_term():
            try:
                if not user_id or not query_embedding: return ""
                results = await long_term_memory_manager.retrieve_memory(
                    user_id=user_id,
                    query=user_query,
                    top_k=5,
                    precomputed_embedding=query_embedding
                )
                if not results: return ""
                
                context_parts = ["### KIẾN THỨC ĐÃ LƯU (Long-term Memory):"]
                for i, res in enumerate(results):
                    entity = res.get("entity_name", "unknown").upper()
                    context_parts.append(f"--- Document {i+1} [{entity}] ---")
                    context_parts.append(res.get("content", ""))
                return "\n".join(context_parts)
            except Exception as e:
                logger.error(f"Error fetching long-term memory: {e}")
                return ""

        # Run parallel with timeout to avoid blocking the whole agent
        try:
            cases_text, long_term_text = await asyncio.gather(
                fetch_cases(), 
                fetch_long_term()
            )
        except Exception as e:
            logger.error(f"Parallel retrieval failed: {e}")
            cases_text, long_term_text = "", ""

        return {
            "retrieved_cases": cases_text or "No examples available.",
            "long_term_context": long_term_text or "No relevant long-term memory found."
        }
