import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.soccer_agent.nodes.memory_saving import SaveToMemoryNode

@pytest.fixture
def memory_saving_node():
    node = SaveToMemoryNode()
    return node

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.memory_saving.asyncio.create_task')
async def test_save_to_memory_node_saves_chat_history(mock_create_task, memory_saving_node):
    """Verify that save_to_memory_node schedules background save for chat history."""
    state = {
        "messages": [
            HumanMessage(content="User query"),
            AIMessage(content="Final response")
        ],
        "tool_calls_history": [],
        "tool_results_history": []
    }
    config = {"metadata": {"thread_id": "test-session"}}

    result = await memory_saving_node.save_to_memory_node(state, config)

    assert result == {}
    # Should have scheduled background save for chat history
    mock_create_task.assert_called()

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.memory_saving.asyncio.create_task')
async def test_save_to_memory_node_no_messages(mock_create_task, memory_saving_node):
    """When no messages exist, no background tasks should be created."""
    state = {
        "messages": [],
        "tool_calls_history": [],
        "tool_results_history": []
    }
    config = {"metadata": {"thread_id": "test-session"}}

    result = await memory_saving_node.save_to_memory_node(state, config)

    assert result == {}
    mock_create_task.assert_not_called()
