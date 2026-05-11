import logging
from langgraph.graph.state import RunnableConfig
from app.schema.soccer_agent.state import AgentState
from app.soccer_agent.case_bank.retriever import CaseBankRetriever
from app.soccer_agent.case_bank.cache import case_bank_cache

logger = logging.getLogger(__name__)

class RetrieveCasesNode:
    def __init__(self, case_bank_retriever: CaseBankRetriever):
        self.case_bank_retriever = case_bank_retriever

    async def retrieve_cases_node(self, state: AgentState, config: RunnableConfig) -> dict:
        """Retrieve few-shot planning examples from case bank (with Redis cache)."""
        messages = state.get("messages", [])
        user_query = messages[-1].content if messages else ""
        query = state.get("claried_query") or user_query
        has_media = bool(state.get("additional_material"))

        cached = case_bank_cache.get(query, has_media)
        if cached:
            logger.info("CaseBankCache: hit")
            return {"retrieved_cases": cached}

        examples = await self.case_bank_retriever.retrieve(query, has_media)
        if examples:
            case_bank_cache.set(query, has_media, examples)
        return {"retrieved_cases": examples or None}
