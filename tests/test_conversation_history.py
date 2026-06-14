import pytest
import uuid
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from app.soccer_agent.nodes.conversation_history import ConversationHistoryNode

@pytest.fixture
def history_node():
    return ConversationHistoryNode()

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.conversation_history.ConversationHistoryManager')
async def test_get_conversational_history_no_history(mock_manager_class, history_node):
    # Setup mock manager
    mock_manager = MagicMock()
    mock_manager.load_messages.return_value = []
    mock_manager_class.return_value = mock_manager
    
    state = {"messages": [HumanMessage(content="Hello")]}
    # thread_id must be a valid UUID string
    thread_uuid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_uuid}}

    result = await history_node.get_conversational_history(state, config)

    assert "conversation_history" in result
    assert result["conversation_history"] == "No conversation history."
    assert "recent_msgs_for_qu" not in result
    mock_manager_class.assert_called_once_with(session_id=thread_uuid)
    mock_manager.load_messages.assert_called_once()

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.conversation_history.ConversationHistoryManager')
async def test_get_conversational_history_with_history(mock_manager_class, history_node):
    # Setup mock manager
    mock_manager = MagicMock()
    mock_manager.load_messages.return_value = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello")
    ]
    mock_manager_class.return_value = mock_manager
    
    state = {"messages": [HumanMessage(content="New query")]}
    thread_uuid = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_uuid}}
    
    result = await history_node.get_conversational_history(state, config)
    
    assert "conversation_history" in result
    assert "### LỊCH SỬ HỘI THOẠI GẦN ĐÂY:" in result["conversation_history"]
    assert "User: Hi" in result["conversation_history"]
    assert "Assistant: Hello" in result["conversation_history"]
    assert "recent_msgs_for_qu" not in result
    mock_manager_class.assert_called_once_with(session_id=thread_uuid)
    mock_manager.load_messages.assert_called_once()

def test_create_user_message_string_text_only():
    msg = HumanMessage(content="Hello there")
    res = ConversationHistoryNode.create_user_message_string(msg)
    assert res == "User: Hello there"

def test_create_user_message_string_with_image():
    msg = HumanMessage(content=[
        {"type": "text", "text": "Look at this player"},
        {"type": "image_url", "image_url": {"url": "https://example.com/sessions/thread1/abcd-1234.jpg"}}
    ])
    res = ConversationHistoryNode.create_user_message_string(msg)
    assert "User: Look at this player" in res
    assert "Attached image: abcd-1234" in res
    assert "Attached video" not in res

def test_create_user_message_string_with_video():
    msg = HumanMessage(content=[
        {"type": "text", "text": "Analyze this clip"},
        {"type": "video_url", "video_url": {"url": "https://example.com/sessions/thread1/xyz-789.mp4"}}
    ])
    res = ConversationHistoryNode.create_user_message_string(msg)
    assert "User: Analyze this clip" in res
    assert "Attached video: xyz-789" in res
    assert "Attached image" not in res

def test_create_user_message_string_with_both():
    msg = HumanMessage(content=[
        {"type": "text", "text": "Compare these"},
        {"type": "image_url", "image_url": {"url": "https://example.com/sessions/thread1/img123.png"}},
        {"type": "video_url", "video_url": {"url": "https://example.com/sessions/thread1/vid456.mp4"}}
    ])
    res = ConversationHistoryNode.create_user_message_string(msg)
    assert "User: Compare these" in res
    assert "Attached image: img123" in res
    assert "Attached video: vid456" in res
