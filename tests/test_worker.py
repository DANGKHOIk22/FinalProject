import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolCall
from langgraph.types import Send

from app.soccer_agent.nodes.worker import WorkerNodes

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
        "need_call_tools": False,
        "tool_chains": [],
        "sub_queries": [],
        "clarified_query": "Hello",
        "messages": [HumanMessage(content="Hello")]
    }
    config = {}

    result = worker_nodes.trigger_workers(state, config)
    assert result == "aggregator"

def test_trigger_workers_with_tools(worker_nodes):
    state = {
        "need_call_tools": True,
        "tool_chains": [["tool1"], ["tool2"]],
        "sub_queries": ["q1", "q2"],
        "clarified_query": "Hello",
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
@patch('app.soccer_agent.nodes.worker.sub_query_cache')
async def test_check_cache_node_hit(mock_cache, worker_nodes):
    mock_cache.check.return_value = "Cached Answer"

    state = {"sub_query": "q1", "additional_material": {}, "tool_chain": ["entity_augment"]}
    result = await worker_nodes._check_cache_node(state, {})

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
    
    async def fake_to_thread(fn, *args, **kwargs):
        fn(*args, **kwargs)

    with patch('app.soccer_agent.nodes.worker.sub_query_cache') as mock_cache:
        with patch('app.soccer_agent.nodes.worker.asyncio.to_thread', side_effect=fake_to_thread):
            result = await worker_nodes._execution_node(state, config)
            # Yield to event loop so create_task(fake_to_thread(...)) runs
            import asyncio as _asyncio
            await _asyncio.sleep(0)

        assert "worker_result" in result
        assert result["worker_result"] == ["Final Answer"]
        assert len(result["messages"]) == 2  # Added system prompt or human prompt + AIMessage
        assert isinstance(result["messages"][-1], AIMessage)
        # Successful result must be cached
        mock_cache.set.assert_called_once_with("q1", "Final Answer", {}, None)

@pytest.mark.asyncio
async def test_execution_node_error_not_cached(worker_nodes, mock_execution_llm):
    """Error responses must never be written to the semantic cache."""
    mock_execution_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

    state = {
        "sub_query": "q1",
        "tool_chain": ["tool1"],
        "messages": [],
        "tool_calls_history": [],
        "tool_results_history": []
    }
    config = {}

    with patch('app.soccer_agent.nodes.worker.sub_query_cache') as mock_cache:
        result = await worker_nodes._execution_node(state, config)

        assert result["worker_result"] == [
            "The execution has been stopped due to an error. Please try again later."
        ]
        mock_cache.set.assert_not_called()

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

@pytest.mark.asyncio
@patch("app.soccer_agent.nodes.worker.adispatch_custom_event", new_callable=AsyncMock)
async def test_tool_node_dispatches_custom_tool_events(mock_dispatch, worker_nodes):
    worker_nodes._tool_executor = MagicMock()
    worker_nodes._tool_executor.ainvoke = AsyncMock(return_value={"messages": []})

    state = {
        "worker_index": 1,
        "worker_id": "worker-1",
        "sub_query": "latest injuries",
        "tool_chain": ["web_news_search"],
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "web_news_search", "args": {"query": "latest injuries"}, "id": "call-1"}],
            )
        ],
    }
    config = {"configurable": {"thread_id": "thread-1"}}

    result = await worker_nodes._tool_node(state, config)

    assert result == {"messages": []}
    event_names = [call.args[0] for call in mock_dispatch.await_args_list]
    assert event_names == ["parallel_tool_started", "parallel_tool_finished"]

    started_payload = mock_dispatch.await_args_list[0].kwargs["data"]
    finished_payload = mock_dispatch.await_args_list[1].kwargs["data"]
    assert started_payload["worker_id"] == "worker-1"
    assert started_payload["tool_name"] == "web_news_search"
    assert started_payload["tool_call_id"] == "call-1"
    assert finished_payload["tool_call_id"] == "call-1"
