import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from app.soccer_agent.nodes.tool_planning import ToolChainPlanningNode
from app.schema.soccer_agent.state import PlanningOutput

@pytest.fixture
def mock_planning_llm():
    llm = MagicMock()
    # Mock ainvoke to return an AIMessage-like object
    mock_response = MagicMock()
    mock_response.text = '{"need_call_tools": true, "tool_chains": [["tool1"]], "sub_queries": ["query1"]}'
    llm.ainvoke = AsyncMock(return_value=mock_response)
    return llm

@pytest.fixture
def mock_parser():
    parser = MagicMock(spec=PydanticOutputParser)
    parser.get_format_instructions.return_value = "Format instructions here"
    # Mock parsing behavior
    parser.parse.return_value = PlanningOutput(
        need_call_tools=True,
        tool_chains=[["tool1"]],
        sub_queries=["query1"]
    )
    return parser

@pytest.fixture
def planning_node(mock_planning_llm, mock_parser):
    return ToolChainPlanningNode(
        planning_llm=mock_planning_llm,
        planning_parser=mock_parser,
        tools=[]
    )

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.tool_planning.get_client')
@patch('app.soccer_agent.nodes.tool_planning.adispatch_custom_event')
async def test_tool_chain_planning(mock_dispatch, mock_get_client, planning_node):
    # Mock langfuse client
    mock_langfuse = MagicMock()
    mock_get_client.return_value = mock_langfuse
    mock_langfuse.start_as_current_observation.return_value.__enter__.return_value = MagicMock()

    state = {
        "messages": [HumanMessage(content="test query")],
        "claried_query": "test query",
        "additional_material": [],
        "conversation_history": "History"
    }
    config = {"metadata": {}}
    
    result = await planning_node.tool_chain_planning(state, config)
    
    assert "planning_output" in result
    assert result["planning_output"].need_call_tools is True
    assert result["planning_output"].tool_chains == [["tool1"]]
    planning_node.planning_llm.ainvoke.assert_called_once()
    planning_node.planning_parser.parse.assert_called_once()
