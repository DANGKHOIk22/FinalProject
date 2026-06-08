import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage
from app.soccer_agent.nodes.context_retrieval import ContextRetrievalNode
from app.soccer_agent.case_bank.retriever import CaseBankRetriever

@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=CaseBankRetriever)
    retriever.retrieve = AsyncMock(return_value="example1\nexample2")
    return retriever

@pytest.fixture
def context_node(mock_retriever):
    return ContextRetrievalNode(case_bank_retriever=mock_retriever)


@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.context_retrieval.context_cache')
@patch('app.soccer_agent.nodes.context_retrieval.long_term_memory_manager')
async def test_context_retrieval_cache_hit(mock_ltm, mock_sem_cache, context_node):
    """When semantic cache hits, return cached result directly."""
    import json
    cached = {"retrieved_cases": "cached_case", "long_term_context": "cached_lt"}
    mock_sem_cache.check.return_value = json.dumps(cached)

    state = {"messages": [HumanMessage(content="test query")], "user_query": "test query", "additional_material": {}}
    config = {"metadata": {"thread_id": "t1"}}

    result = await context_node.retrieve_context_node(state, config)

    assert result == cached
    mock_sem_cache.check.assert_called_once_with("test query")
    # Should NOT call retriever or LTM on cache hit
    context_node.case_bank_retriever.retrieve.assert_not_called()


@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.context_retrieval.context_cache')
@patch('app.soccer_agent.nodes.context_retrieval.long_term_memory_manager')
async def test_context_retrieval_cache_miss(mock_ltm, mock_sem_cache, context_node):
    """When semantic cache misses, fetch from case bank + long-term memory."""
    mock_sem_cache.check.return_value = None
    mock_ltm._get_embedding = AsyncMock(return_value=[0.1] * 768)
    mock_ltm.retrieve_memory = AsyncMock(return_value=[])

    context_node.case_bank_retriever.retrieve = AsyncMock(return_value="example_new")

    state = {"messages": [HumanMessage(content="test query")], "user_query": "test query", "additional_material": {}}
    config = {"metadata": {"thread_id": "t1"}}

    result = await context_node.retrieve_context_node(state, config)

    assert "retrieved_cases" in result
    assert result["retrieved_cases"] == "example_new"
    context_node.case_bank_retriever.retrieve.assert_called_once()
