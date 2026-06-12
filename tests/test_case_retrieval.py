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
@patch('app.soccer_agent.nodes.context_retrieval.case_bank_cache')
@patch('app.soccer_agent.nodes.context_retrieval.long_term_memory_manager')
async def test_context_retrieval_case_cache_hit(mock_ltm, mock_cb_cache, context_node):
    """When case bank cache hits, skip the retriever but still fetch long-term memory."""
    mock_cb_cache.get.return_value = "cached_case"
    mock_ltm._get_embedding = AsyncMock(return_value=[0.1] * 768)
    mock_ltm.retrieve_memory = AsyncMock(return_value=[])

    state = {"messages": [HumanMessage(content="test query")], "user_query": "test query", "additional_material": []}
    config = {"metadata": {"thread_id": "t1"}}

    result = await context_node.retrieve_context_node(state, config)

    assert result["retrieved_cases"] == "cached_case"
    mock_cb_cache.get.assert_called_once_with("test query", False)
    # Should NOT call retriever or re-save cache on cache hit
    context_node.case_bank_retriever.retrieve.assert_not_called()
    mock_cb_cache.set.assert_not_called()
    # Long-term memory is always fetched fresh (per-user data, never cached)
    mock_ltm.retrieve_memory.assert_called_once()


@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.context_retrieval.case_bank_cache')
@patch('app.soccer_agent.nodes.context_retrieval.long_term_memory_manager')
async def test_context_retrieval_case_cache_miss(mock_ltm, mock_cb_cache, context_node):
    """When case bank cache misses, fetch from retriever and save to cache."""
    mock_cb_cache.get.return_value = None
    mock_ltm._get_embedding = AsyncMock(return_value=[0.1] * 768)
    mock_ltm.retrieve_memory = AsyncMock(return_value=[])

    context_node.case_bank_retriever.retrieve = AsyncMock(return_value="example_new")

    state = {"messages": [HumanMessage(content="test query")], "user_query": "test query", "additional_material": []}
    config = {"metadata": {"thread_id": "t1"}}

    result = await context_node.retrieve_context_node(state, config)

    assert result["retrieved_cases"] == "example_new"
    context_node.case_bank_retriever.retrieve.assert_called_once_with("test query", False)
    mock_cb_cache.set.assert_called_once_with("test query", False, "example_new")


@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.context_retrieval.case_bank_cache')
@patch('app.soccer_agent.nodes.context_retrieval.long_term_memory_manager')
async def test_context_retrieval_has_media_passed_to_cache(mock_ltm, mock_cb_cache, context_node):
    """has_media must be part of both cache lookup and save."""
    mock_cb_cache.get.return_value = None
    mock_ltm._get_embedding = AsyncMock(return_value=[0.1] * 768)
    mock_ltm.retrieve_memory = AsyncMock(return_value=[])

    state = {
        "messages": [HumanMessage(content="test query")],
        "user_query": "test query",
        "additional_material": ["http://example.com/image.png"],
    }
    config = {"metadata": {"thread_id": "t1"}}

    await context_node.retrieve_context_node(state, config)

    mock_cb_cache.get.assert_called_once_with("test query", True)
    mock_cb_cache.set.assert_called_once_with("test query", True, "example1\nexample2")
