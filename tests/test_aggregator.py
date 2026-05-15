import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from app.soccer_agent.nodes.aggregator import AggregatorNode

@pytest.fixture
def mock_aggregator_llm():
    llm = MagicMock()
    mock_response = AIMessage(content="Aggregated Final Response")
    llm.ainvoke = AsyncMock(return_value=mock_response)
    return llm

@pytest.fixture
def aggregator_node(mock_aggregator_llm):
    return AggregatorNode(aggregator_llm=mock_aggregator_llm)

@pytest.mark.asyncio
async def test_aggregator_node(aggregator_node, mock_aggregator_llm):
    state = {
        "messages": [HumanMessage(content="Hello")],
        "claried_query": "Hello",
        "parallel_results": ["Result 1", "Result 2"],
        "conversation_history": "History"
    }
    config = {}
    
    result = await aggregator_node.aggregator_node(state, config)
    
    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Aggregated Final Response"
    
    # Verify the prompt was invoked
    mock_aggregator_llm.ainvoke.assert_called_once()
