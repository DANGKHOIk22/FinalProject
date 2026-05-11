import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage
from app.soccer_agent.nodes.conversation_history import ConversationHistoryNode
from app.soccer_agent.memory.session_memory import SessionMemoryManager

@pytest.fixture
def mock_session_memory_manager():
    manager = MagicMock(spec=SessionMemoryManager)
    manager.count_tokens = MagicMock(return_value=100) # Mock small history
    manager.format_compressed_history = MagicMock(return_value="Mocked History")
    manager.load_cached_memory = AsyncMock(return_value=None)
    manager.background_summarise = AsyncMock()
    return manager

@pytest.fixture
def history_node(mock_session_memory_manager):
    node = ConversationHistoryNode(session_memory_manager=mock_session_memory_manager)
    # Mock database retrieval so we don't connect to real postgres
    node.get_memory = AsyncMock()
    node.cleanup = AsyncMock()
    return node

@pytest.mark.asyncio
async def test_get_conversational_history_no_history(history_node):
    # Setup mock to return no history
    mock_memory_object = MagicMock()
    mock_memory_object.load_memory_variables.return_value = {"history": []}
    history_node.get_memory.return_value = (mock_memory_object, MagicMock(), MagicMock())
    
    state = {"messages": [HumanMessage(content="Hello")]}
    config = {"metadata": {"thread_id": "test-session"}}
    
    result = await history_node.get_conversational_history(state, config)
    
    assert "conversation_history" in result
    assert result["conversation_history"] == ""
    assert result["recent_msgs_for_qu"] == []
    history_node.get_memory.assert_called_once_with(session_id="test-session")

@pytest.mark.asyncio
async def test_get_conversational_history_with_history(history_node, mock_session_memory_manager):
    mock_memory_object = MagicMock()
    mock_memory_object.load_memory_variables.return_value = {
        "history": [HumanMessage(content="Hi"), HumanMessage(content="Hello")]
    }
    history_node.get_memory.return_value = (mock_memory_object, MagicMock(), MagicMock())
    
    state = {"messages": [HumanMessage(content="New query")]}
    config = {"metadata": {"thread_id": "test-session"}}
    
    result = await history_node.get_conversational_history(state, config)
    
    assert "conversation_history" in result
    assert "LỊCH SỬ HỘI THOẠI" in result["conversation_history"]
    assert len(result["recent_msgs_for_qu"]) == 2
