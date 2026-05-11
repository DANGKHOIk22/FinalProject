import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.soccer_agent.nodes.memory_saving import SaveToMemoryNode
from app.soccer_agent.memory.session_memory import SessionMemoryManager

@pytest.fixture
def mock_session_memory_manager():
    manager = MagicMock(spec=SessionMemoryManager)
    manager.add_message_to_cache = AsyncMock()
    return manager

@pytest.fixture
def memory_saving_node(mock_session_memory_manager):
    return SaveToMemoryNode(session_memory_manager=mock_session_memory_manager)

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
    
    result = await memory_saving_node.save_to_memory_node(state, config)
    
    assert result == {}  # Should return empty dict
    
    # We shouldn't await create_task in test directly easily, but we can verify it was called
    mock_create_task.assert_called()
