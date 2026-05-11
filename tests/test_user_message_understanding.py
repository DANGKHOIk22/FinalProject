import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END

from app.soccer_agent.nodes.user_message_understanding import UnderstandUserMessageNode
from app.soccer_agent.memory.query_understanding import QueryUnderstandingPipeline
from app.schema.soccer_agent.state import AgentState

@pytest.fixture
def mock_query_understanding():
    pipeline = MagicMock(spec=QueryUnderstandingPipeline)
    return pipeline

@pytest.fixture
def understanding_node(mock_query_understanding):
    return UnderstandUserMessageNode(query_understanding=mock_query_understanding)

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.user_message_understanding.adispatch_custom_event')
async def test_understand_user_message_clear_query(mock_dispatch, understanding_node, mock_query_understanding):
    # Setup mock to return clear query
    mock_output = MagicMock()
    mock_output.is_ambiguous = False
    mock_output.clarified_query = "Who is Messi?"
    mock_query_understanding.run = AsyncMock(return_value=mock_output)
    
    state = {"messages": [HumanMessage(content="Who is he?")], "effective_memory": None, "recent_msgs_for_qu": []}
    config = {}
    
    result = await understanding_node.understand_user_message(state, config)
    
    assert "claried_query" in result
    assert result["claried_query"] == "Who is Messi?"

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.user_message_understanding.adispatch_custom_event')
async def test_understand_user_message_ambiguous_query(mock_dispatch, understanding_node, mock_query_understanding):
    # Setup mock to return ambiguous query
    mock_output = MagicMock()
    mock_output.is_ambiguous = True
    mock_output.clarifying_questions = ["Who are you referring to?"]
    mock_query_understanding.run = AsyncMock(return_value=mock_output)
    
    state = {"messages": [HumanMessage(content="Who is he?")], "additional_material": []}
    config = {}
    
    result = await understanding_node.understand_user_message(state, config)
    
    assert isinstance(result, Command)
    assert result.goto == END
    
    update_dict = result.update
    assert "messages" in update_dict
    assert isinstance(update_dict["messages"][-1], AIMessage)
    assert "Bạn có thể làm rõ thêm không" in update_dict["messages"][-1].content
