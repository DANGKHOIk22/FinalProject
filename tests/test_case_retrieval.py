import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage
from app.soccer_agent.nodes.case_retrieval import RetrieveCasesNode
from app.soccer_agent.case_bank.retriever import CaseBankRetriever

@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=CaseBankRetriever)
    return retriever

@pytest.fixture
def case_retrieval_node(mock_retriever):
    return RetrieveCasesNode(case_bank_retriever=mock_retriever)

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.case_retrieval.case_bank_cache')
async def test_retrieve_cases_node_cache_hit(mock_cache, case_retrieval_node):
    # Mock cache hit
    mock_cache.get.return_value = ["example1", "example2"]
    
    state = {"messages": [HumanMessage(content="test query")], "claried_query": "test query", "additional_material": []}
    config = {}
    
    result = await case_retrieval_node.retrieve_cases_node(state, config)
    
    assert "retrieved_cases" in result
    assert result["retrieved_cases"] == ["example1", "example2"]
    mock_cache.get.assert_called_once_with("test query", False)
    case_retrieval_node.case_bank_retriever.retrieve.assert_not_called()

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.case_retrieval.case_bank_cache')
async def test_retrieve_cases_node_cache_miss(mock_cache, case_retrieval_node):
    # Mock cache miss
    mock_cache.get.return_value = None
    
    # Mock retriever
    case_retrieval_node.case_bank_retriever.retrieve = AsyncMock(return_value=["example_new"])
    
    state = {"messages": [HumanMessage(content="test query")]}
    config = {}
    
    result = await case_retrieval_node.retrieve_cases_node(state, config)
    
    assert "retrieved_cases" in result
    assert result["retrieved_cases"] == ["example_new"]
    mock_cache.set.assert_called_once_with("test query", False, ["example_new"])
