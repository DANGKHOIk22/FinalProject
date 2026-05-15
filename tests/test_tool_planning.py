import pytest
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
    json_content = '{"clarified_query": "test query clarified", "is_ambiguous": false, "clarifying_questions": [], "need_call_tools": true, "tool_chains": [["tool1"]], "sub_queries": ["query1"]}'
    
    # Configure the mock response to return the JSON string for both content and text
    mock_response.content = json_content
    mock_response.text = json_content
    
    llm.ainvoke = AsyncMock(return_value=mock_response)
    return llm

@pytest.fixture
def planning_node(mock_planning_llm):
    return UnifiedPlanningNode(
        planning_llm=mock_planning_llm,
        tools=[]
    )

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.unified_planning.adispatch_custom_event')
async def test_unified_planning(mock_dispatch, planning_node):
    state = {
        "messages": [HumanMessage(content="test query")],
        "additional_material": [],
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
    assert result["planning_output"].tool_chains == [["tool1"]]
    assert result["clarified_query"] == "test query clarified"
    planning_node.planning_llm.ainvoke.assert_called_once()
