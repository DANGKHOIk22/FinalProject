import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage
from app.soccer_agent.nodes.unified_planning import UnifiedPlanningNode
from app.schema.soccer_agent.state import UnifiedPlanningOutput

@pytest.fixture
def mock_planning_llm():
    llm = MagicMock()
    # Mock ainvoke to return an AIMessage-like object
    mock_response = MagicMock()
    # PydanticOutputParser looks for a string. 
    # We need to ensure response.content or response.text is a valid JSON string.
    json_content = '{"clarified_query": "test query clarified", "need_call_tools": true, "planned_chains": [{"chain": ["tool1"], "sub_query": "query1", "confidence": 0.95, "is_ambiguous": false, "clarifying_question": null}]}'

    mock_response.content = json_content
    mock_response.text = json_content
    
    llm.ainvoke = AsyncMock(return_value=mock_response)
    return llm

@pytest.fixture
def planning_node(mock_planning_llm):
    return UnifiedPlanningNode(
        planning_llm=mock_planning_llm,
        planning_multimodal_llm=mock_planning_llm,
        tools=[]
    )

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.unified_planning.adispatch_custom_event')
async def test_unified_planning(mock_dispatch, planning_node):
    state = {
        "messages": [HumanMessage(content="test query")],
        "additional_material": {},
        "conversation_history": "History",
        "long_term_context": "Long term context",
        "retrieved_cases": "Examples",
        "time_context": "2024-01-01"
    }
    config = {"metadata": {"thread_id": "test_thread"}}
    
    result = await planning_node.unified_planning_node(state, config)
    
    # If parsing fails, it returns a fallback dict without "planning_output"
    # So we check if it succeeded
    assert "planning_output" in result, f"Planning failed. Result was: {result}"
    assert result["planning_output"].need_call_tools is True
    assert result["tool_chains"] == [["tool1"]]
    assert result["sub_queries"] == ["query1"]
    assert result["clarified_query"] == "test query clarified"
    assert result["pending_clarifications"] == []
    planning_node.planning_llm.ainvoke.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for extract_media_uuids static method
# ---------------------------------------------------------------------------

def test_extract_media_uuids_with_registry():
    mock_registry = MagicMock()
    mock_registry.extract_uuids_from_message.return_value = {
        "image_ids": ["img_abc", "img_def"],
        "video_id": "vid_xyz"
    }
    
    config = {"configurable": {"media_registry": mock_registry}}
    messages = [HumanMessage(content="Describe these media files")]
    additional_material = {}
    
    res = UnifiedPlanningNode.extract_media_uuids(messages, config, additional_material)
    
    assert res["image_id"] == ["img_abc", "img_def"]
    assert res["video_id"] == "vid_xyz"
    mock_registry.extract_uuids_from_message.assert_called_once_with(messages[-1])

def test_extract_media_uuids_no_registry():
    config = {"configurable": {}}
    messages = [HumanMessage(content="Describe these media files")]
    additional_material = {"image_id": ["old_img"], "video_id": "old_vid"}
    
    res = UnifiedPlanningNode.extract_media_uuids(messages, config, additional_material)
    
    assert res["image_id"] == []
    assert res["video_id"] is None

def test_extract_media_uuids_empty_messages():
    mock_registry = MagicMock()
    config = {"configurable": {"media_registry": mock_registry}}
    messages = []
    additional_material = {}
    
    res = UnifiedPlanningNode.extract_media_uuids(messages, config, additional_material)
    
    assert res["image_id"] == []
    assert res["video_id"] is None
    mock_registry.extract_uuids_from_message.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for extract_game_context static method
# ---------------------------------------------------------------------------

def test_extract_game_context_state_only():
    state = {"video_current_time": 42.5}
    additional_material = {"game_id": "match_001"}
    
    game_id, current_time = UnifiedPlanningNode.extract_game_context(state, additional_material)
    
    assert game_id == "match_001"
    assert current_time == 42.5
    assert additional_material["game_id"] == "match_001"

def test_extract_game_context_copilotkit_dict():
    state = {
        "video_current_time": 10.0,
        "copilotkit": {
            "context": [
                {
                    "value": {
                        "game_id": "match_copilot_1",
                        "current_time": 105.0
                    }
                }
            ]
        }
    }
    additional_material = {}
    
    game_id, current_time = UnifiedPlanningNode.extract_game_context(state, additional_material)
    
    assert game_id == "match_copilot_1"
    assert current_time == 105.0
    assert additional_material["game_id"] == "match_copilot_1"

def test_extract_game_context_copilotkit_json_str():
    state = {
        "video_current_time": 20.0,
        "copilotkit": {
            "context": [
                MagicMock(value='{"game_id": "match_copilot_2", "current_time": 200.0}')
            ]
        }
    }
    additional_material = {}
    
    game_id, current_time = UnifiedPlanningNode.extract_game_context(state, additional_material)
    
    assert game_id == "match_copilot_2"
    assert current_time == 200.0
    assert additional_material["game_id"] == "match_copilot_2"

def test_extract_game_context_copilotkit_invalid_json():
    state = {
        "video_current_time": 30.0,
        "copilotkit": {
            "context": [
                {"value": "{invalid json}"}
            ]
        }
    }
    additional_material = {"game_id": "fallback_game"}
    
    game_id, current_time = UnifiedPlanningNode.extract_game_context(state, additional_material)
    
    assert game_id == "fallback_game"
    assert current_time == 30.0
    assert additional_material["game_id"] == "fallback_game"


# ---------------------------------------------------------------------------
# Tests for remove_video_content static method
# ---------------------------------------------------------------------------

def test_remove_video_content_string():
    msg = HumanMessage(content="Hello world")
    res = UnifiedPlanningNode.remove_video_content(msg)
    assert res.content == "Hello world"

def test_remove_video_content_list_no_video():
    content = [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "https://example.com/sessions/thread1/img.png"}}
    ]
    msg = HumanMessage(content=content)
    res = UnifiedPlanningNode.remove_video_content(msg)
    assert len(res.content) == 2
    assert res.content[0]["type"] == "text"
    assert res.content[1]["type"] == "image_url"

def test_remove_video_content_list_with_video():
    content = [
        {"type": "text", "text": "Compare these files"},
        {"type": "image_url", "image_url": {"url": "https://example.com/sessions/thread1/img.png"}},
        {"type": "video_url", "video_url": {"url": "https://example.com/sessions/thread1/vid.mp4"}}
    ]
    msg = HumanMessage(content=content)
    res = UnifiedPlanningNode.remove_video_content(msg)
    
    # Video url block should be removed
    assert len(res.content) == 2
    assert res.content[0]["type"] == "text"
    assert res.content[1]["type"] == "image_url"
    for part in res.content:
        assert part.get("type") != "video_url"
