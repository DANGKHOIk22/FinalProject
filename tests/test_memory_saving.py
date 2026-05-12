import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.soccer_agent.nodes.memory_saving import SaveToMemoryNode

@pytest.fixture
def memory_saving_node():
    node = SaveToMemoryNode()
    node.get_memory = AsyncMock()
    return node

def test_build_tool_summary_for_memory(memory_saving_node):
    tool_calls = [
        {"name": "tool1", "args": {"query": "hello"}, "id": "1", "type": "tool_call"},
        {"name": "tool2", "args": {"filter": "recent"}, "id": "2", "type": "tool_call"}
    ]
    tool_messages = [
        ToolMessage(content="Result 1", name="tool1", tool_call_id="1"),
        ToolMessage(content="Result 2", name="tool2", tool_call_id="2")
    ]
    
    summary = memory_saving_node._build_tool_summary_for_memory(tool_calls, tool_messages, "Final")
    
    assert "tool1" in summary
    assert "Result 1" in summary
    assert "tool2" in summary
    assert "Result 2" in summary

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.memory_saving.asyncio.create_task')
async def test_save_to_memory_node(mock_create_task, memory_saving_node):
    state = {
        "messages": [
            HumanMessage(content="User query"),
            AIMessage(content="Final response")
        ],
        "tool_results_history": [
            ToolMessage(content="Tool result", name="test_tool", tool_call_id="1")
        ]
    }
    config = {"metadata": {"thread_id": "test-session"}}
    
    # Mock get_memory to avoid postgres connection
    memory_saving_node.get_memory = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    
    result = await memory_saving_node.save_to_memory_node(state, config)
    
    assert result == {}  # Should return empty dict
    mock_create_task.assert_called()
