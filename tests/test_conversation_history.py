import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from app.soccer_agent.nodes.conversation_history import ConversationHistoryNode

@pytest.fixture
def history_node():
    node = ConversationHistoryNode()
    # Mock database retrieval so we don't connect to real postgres
    node.get_memory = AsyncMock()
    return node

@pytest.mark.asyncio
async def test_get_conversational_history_no_history(history_node):
    # Setup mock to return no history
    mock_memory_object = MagicMock()
    mock_memory_object.load_memory_variables.return_value = {"history": []}
    history_node.get_memory.return_value = (mock_memory_object, MagicMock(), MagicMock())
    
    state = {"messages": [HumanMessage(content="Hello")]}
    config = {"configurable": {"thread_id": "test-session"}}

    result = await history_node.get_conversational_history(state, config)

    assert "conversation_history" in result
    assert result["conversation_history"] == ""
    assert result["recent_msgs_for_qu"] == []
    history_node.get_memory.assert_called_once_with(session_id="test-session")

@pytest.mark.asyncio
async def test_get_conversational_history_with_history(history_node):
    mock_memory_object = MagicMock()
    mock_memory_object.load_memory_variables.return_value = {
        "history": [HumanMessage(content="Hi"), AIMessage(content="Hello")]
    }
    history_node.get_memory.return_value = (mock_memory_object, MagicMock(), MagicMock())
    
    state = {"messages": [HumanMessage(content="New query")]}
    config = {"configurable": {"thread_id": "test-session"}}
    
    result = await history_node.get_conversational_history(state, config)
    
    assert "conversation_history" in result
    assert "LỊCH SỬ HỘI THOẠI" in result["conversation_history"]
    assert len(result["recent_msgs_for_qu"]) == 2
