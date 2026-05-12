import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolCall
from langgraph.types import Send

from app.soccer_agent.nodes.worker import WorkerNodes
from app.schema.soccer_agent.state import UnifiedPlanningOutput

@pytest.fixture
def mock_execution_llm():
    llm = MagicMock()
    return llm

@pytest.fixture
def worker_nodes(mock_execution_llm):
    # Mock _build_worker_graph to avoid actually compiling LangGraph during unit tests of the methods
    with patch.object(WorkerNodes, '_build_worker_graph', return_value=MagicMock()):
        return WorkerNodes(execution_llm_with_tools=mock_execution_llm, tools=[])

def test_trigger_workers_no_tools(worker_nodes):
    state = {
        "planning_output": UnifiedPlanningOutput(
            clarified_query="Hello",
            is_ambiguous=False,
            clarifying_questions=[],
            need_call_tools=False,
            tool_chains=[],
            sub_queries=[]
        ),
        "messages": [HumanMessage(content="Hello")]
    }
    config = {}
    
    result = worker_nodes.trigger_workers(state, config)
    assert result == "aggregator_node"

def test_trigger_workers_with_tools(worker_nodes):
    state = {
        "planning_output": UnifiedPlanningOutput(
            clarified_query="Hello",
            is_ambiguous=False,
            clarifying_questions=[],
            need_call_tools=True, 
            tool_chains=[["tool1"], ["tool2"]],
            sub_queries=["q1", "q2"]
        ),
        "messages": [HumanMessage(content="Hello")]
    }
    config = {}
    
    result = worker_nodes.trigger_workers(state, config)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], Send)
    assert result[0].node == "worker_graph"
    assert result[0].arg["sub_query"] == "q1"

@pytest.mark.asyncio
@patch('app.soccer_agent.nodes.worker.semantic_cache')
async def test_check_cache_node_hit(mock_cache, worker_nodes):
    mock_cache.check.return_value = "Cached Answer"
    
    state = {"sub_query": "q1", "additional_material": []}
    result = worker_nodes._check_cache_node(state)
    
    assert "worker_result" in result
    assert result["worker_result"] == ["Cached Answer"]

@pytest.mark.asyncio
async def test_execution_node_returns_text(worker_nodes, mock_execution_llm):
    # Mock LLM to return text without tool calls
    mock_response = AIMessage(content="Final Answer")
    mock_execution_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    state = {
        "sub_query": "q1",
        "tool_chain": ["tool1"],
        "messages": [],
        "tool_calls_history": [],
        "tool_results_history": []
    }
    config = {}
    
    with patch('app.soccer_agent.nodes.worker.semantic_cache') as mock_cache:
        result = await worker_nodes._execution_node(state, config)
        
        assert "worker_result" in result
        assert result["worker_result"] == ["Final Answer"]
        assert len(result["messages"]) == 2  # Added system prompt or human prompt + AIMessage
        assert isinstance(result["messages"][-1], AIMessage)

def test_should_execute_worker(worker_nodes):
    # If worker_result exists (cache hit), it should return "end"
    assert worker_nodes.should_execute_worker({"worker_result": ["Answer"]}) == "end"
    # Otherwise "execute"
    assert worker_nodes.should_execute_worker({}) == "execute"

def test_should_continue_call_tool(worker_nodes):
    # If last message has tool calls, continue
    msg_with_tool = AIMessage(content="", tool_calls=[{"name": "tool1", "args": {}, "id": "1"}])
    assert worker_nodes.should_continue_call_tool({"messages": [msg_with_tool]}) == "continue"
    # Otherwise end
    msg_no_tool = AIMessage(content="Final")
    assert worker_nodes.should_continue_call_tool({"messages": [msg_no_tool]}) == "end"
