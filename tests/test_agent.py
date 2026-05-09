"""Unit tests for SoccerAgent methods.

All external dependencies (LLMs, databases, Langfuse, graphs, copilotkit) are
mocked so these tests run fully offline and instantaneously.

Covered methods:
  - _build_tool_summary_for_memory  (pure)
  - should_execute_worker           (pure)
  - should_continue_call_tool       (pure)
  - _check_cache_node               (sync, mocks semantic_cache)
  - _trigger_workers                (sync, mocks PlanningOutput)
  - _retrieve_cases_node            (async, mocks case_bank_cache + retriever)
  - _aggregator_node                (async, mocks aggregator LLM)
  - _worker_node                    (async, mocks worker_graph)
  - _execution_node                 (async, mocks execution LLM + semantic_cache)
"""
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# Stub out optional packages that may not be installed in the test venv
# ---------------------------------------------------------------------------

def _stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# copilotkit / ag_ui_langgraph — used only by state.py at import time
class _CopilotKitState:
    pass

_stub_module("copilotkit", CopilotKitState=_CopilotKitState)
_stub_module("ag_ui_langgraph")
# langfuse — may be absent; stub the parts used by agent.py
_langfuse_mod = _stub_module(
    "langfuse",
    get_client=MagicMock(return_value=MagicMock()),
    observe=lambda f: f,  # no-op decorator
)

from app.schema.soccer_agent.state import PlanningOutput  # noqa: E402
from app.soccer_agent.agent import SoccerAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture — bypasses __init__ entirely
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """Return a SoccerAgent with all heavy attrs replaced by MagicMocks."""
    with patch.object(SoccerAgent, "__init__", lambda self, **_: None):
        a = SoccerAgent()
    a.planning_llm = MagicMock()
    a.execution_llm = MagicMock()
    a.aggregator_llm = MagicMock()
    a.planning_parser = MagicMock()
    a.checkpointer = None
    a.case_bank_retriever = MagicMock()
    a.session_memory_manager = MagicMock()
    a.query_understanding = MagicMock()
    a.tools = [MagicMock(name="tool_a"), MagicMock(name="tool_b")]
    a.tool_registry = {}
    a.execution_llm_with_tools = MagicMock()
    a.worker_graph = MagicMock()
    a.graph = MagicMock()
    return a


# ---------------------------------------------------------------------------
# _build_tool_summary_for_memory
# ---------------------------------------------------------------------------

class TestBuildToolSummaryForMemory:

    def test_no_tool_calls_returns_final_response_only(self, agent):
        result = agent._build_tool_summary_for_memory([], [], "Final answer.")
        assert result == "Final answer."

    def test_single_tool_call_no_result_produces_tool_usage_block(self, agent):
        tool_call = {"name": "game_search", "args": {"query": "Chelsea vs Burnley"}}
        result = agent._build_tool_summary_for_memory([tool_call], [], "Response.")
        assert "[Tool Usage]" in result
        assert "Step 1" in result
        assert "game_search" in result
        assert "Chelsea vs Burnley" in result
        assert "[Final Response]" in result
        assert "Response." in result

    def test_tool_call_with_result_shows_response_and_artifact(self, agent):
        tool_call = {"name": "game_info_retrieval", "args": {"artifact": "game123"}}
        msg = MagicMock(spec=ToolMessage)
        msg.content = "Match info retrieved."
        msg.artifact = "england_epl/game123"
        result = agent._build_tool_summary_for_memory([tool_call], [msg], "Done.")
        assert "Match info retrieved." in result
        assert "england_epl/game123" in result
        assert "Artifact" in result

    def test_tool_call_result_with_none_artifact_no_artifact_line(self, agent):
        tool_call = {"name": "textual_entity_search", "args": {"entity": "Messi"}}
        msg = MagicMock(spec=ToolMessage)
        msg.content = "Entity found."
        msg.artifact = None
        result = agent._build_tool_summary_for_memory([tool_call], [msg], "Done.")
        # artifact is None → the `if artifact is not None` branch is skipped
        assert "Artifact" not in result

    def test_tool_call_result_missing_artifact_attr_no_artifact_line(self, agent):
        tool_call = {"name": "textual_entity_search", "args": {"entity": "Ronaldo"}}
        msg = MagicMock(spec=["content"])  # no 'artifact' attribute
        msg.content = "Entity found."
        result = agent._build_tool_summary_for_memory([tool_call], [msg], "Done.")
        assert "Artifact" not in result

    def test_multiple_tool_calls_appear_as_sequential_steps(self, agent):
        calls = [
            {"name": "game_search", "args": {"team": "Chelsea"}},
            {"name": "game_info_retrieval", "args": {"artifact": "x/y"}},
        ]
        msgs = [
            MagicMock(content="Search result.", artifact=None),
            MagicMock(content="Match info.", artifact="x/y"),
        ]
        result = agent._build_tool_summary_for_memory(calls, msgs, "Final.")
        assert "Step 1" in result
        assert "Step 2" in result
        assert "game_search" in result
        assert "game_info_retrieval" in result

    def test_fewer_results_than_calls_does_not_raise(self, agent):
        calls = [
            {"name": "step_one", "args": {}},
            {"name": "step_two", "args": {}},
        ]
        msgs = [MagicMock(content="Only one result.", artifact=None)]
        # step_two has no corresponding result — should not raise IndexError
        result = agent._build_tool_summary_for_memory(calls, msgs, "Final.")
        assert "step_one" in result
        assert "step_two" in result


# ---------------------------------------------------------------------------
# should_execute_worker
# ---------------------------------------------------------------------------

class TestShouldExecuteWorker:

    def test_returns_end_when_worker_result_present(self, agent):
        assert agent.should_execute_worker({"worker_result": ["cached"]}) == "end"

    def test_returns_execute_when_worker_result_absent(self, agent):
        assert agent.should_execute_worker({}) == "execute"

    def test_returns_execute_when_worker_result_empty_list(self, agent):
        assert agent.should_execute_worker({"worker_result": []}) == "execute"

    def test_returns_execute_when_worker_result_none(self, agent):
        assert agent.should_execute_worker({"worker_result": None}) == "execute"


# ---------------------------------------------------------------------------
# should_continue_call_tool
# ---------------------------------------------------------------------------

class TestShouldContinueCallTool:

    def test_returns_continue_when_last_ai_message_has_tool_calls(self, agent):
        msg = AIMessage(content="", tool_calls=[{"name": "game_search", "id": "1", "args": {}}])
        assert agent.should_continue_call_tool({"messages": [msg]}) == "continue"

    def test_returns_end_when_last_ai_message_has_no_tool_calls(self, agent):
        msg = AIMessage(content="Done.")
        assert agent.should_continue_call_tool({"messages": [msg]}) == "end"

    def test_returns_end_when_messages_list_is_empty(self, agent):
        assert agent.should_continue_call_tool({"messages": []}) == "end"

    def test_returns_end_when_last_message_is_tool_message_not_ai(self, agent):
        # ToolMessage is not AIMessage → isinstance check fails
        msg = ToolMessage(content="result", tool_call_id="abc")
        assert agent.should_continue_call_tool({"messages": [msg]}) == "end"

    def test_returns_end_when_state_has_no_messages_key(self, agent):
        assert agent.should_continue_call_tool({}) == "end"


# ---------------------------------------------------------------------------
# _check_cache_node
# ---------------------------------------------------------------------------

class TestCheckCacheNode:

    def test_returns_cached_result_on_semantic_cache_hit(self, agent):
        with patch("app.soccer_agent.agent.semantic_cache") as mock_cache:
            mock_cache.check.return_value = "Cached answer."
            state = {"sub_query": "Who won?", "additional_material": []}
            result = agent._check_cache_node(state)
        assert result == {"worker_result": ["Cached answer."]}
        mock_cache.check.assert_called_once_with("Who won?", [])

    def test_returns_empty_dict_on_cache_miss(self, agent):
        with patch("app.soccer_agent.agent.semantic_cache") as mock_cache:
            mock_cache.check.return_value = None
            result = agent._check_cache_node({"sub_query": "Who won?", "additional_material": []})
        assert result == {}

    def test_returns_empty_dict_when_sub_query_missing(self, agent):
        # No sub_query key → early return without touching cache
        with patch("app.soccer_agent.agent.semantic_cache") as mock_cache:
            result = agent._check_cache_node({})
        assert result == {}
        mock_cache.check.assert_not_called()

    def test_passes_additional_material_to_cache_check(self, agent):
        with patch("app.soccer_agent.agent.semantic_cache") as mock_cache:
            mock_cache.check.return_value = None
            agent._check_cache_node({"sub_query": "q", "additional_material": ["img.jpg"]})
        mock_cache.check.assert_called_once_with("q", ["img.jpg"])


# ---------------------------------------------------------------------------
# _trigger_workers
# ---------------------------------------------------------------------------

class TestTriggerWorkers:

    def _state(self, planning_output, claried_query="clarified", additional_material=None, messages=None):
        if messages is None:
            msg = MagicMock()
            msg.content = "original"
            messages = [msg]
        return {
            "planning_output": planning_output,
            "claried_query": claried_query,
            "additional_material": additional_material or [],
            "messages": messages,
        }

    def test_returns_aggregator_node_when_need_call_tools_false(self, agent):
        po = PlanningOutput(tool_chains=[["game_search"]], sub_queries=["q"], need_call_tools=False)
        result = agent._trigger_workers(self._state(po), MagicMock())
        assert result == "aggregator_node"

    def test_returns_aggregator_node_when_tool_chains_empty(self, agent):
        po = PlanningOutput(tool_chains=[], sub_queries=[], need_call_tools=True)
        result = agent._trigger_workers(self._state(po), MagicMock())
        assert result == "aggregator_node"

    def test_returns_aggregator_node_when_planning_output_none(self, agent):
        result = agent._trigger_workers(self._state(None), MagicMock())
        assert result == "aggregator_node"

    def test_returns_one_send_per_chain(self, agent):
        po = PlanningOutput(
            tool_chains=[["game_search", "game_info_retrieval"], ["textual_entity_search"]],
            sub_queries=["match result?", "who is Messi?"],
            need_call_tools=True,
        )
        result = agent._trigger_workers(self._state(po), MagicMock())
        assert isinstance(result, list)
        assert len(result) == 2

    def test_send_worker_state_uses_matching_sub_query(self, agent):
        po = PlanningOutput(
            tool_chains=[["game_search"], ["textual_entity_search"]],
            sub_queries=["match Q", "entity Q"],
            need_call_tools=True,
        )
        result = agent._trigger_workers(self._state(po), MagicMock())
        assert result[0].arg["sub_query"] == "match Q"
        assert result[1].arg["sub_query"] == "entity Q"

    def test_send_uses_claried_query_when_sub_queries_shorter_than_chains(self, agent):
        po = PlanningOutput(
            tool_chains=[["game_search"], ["textual_entity_search"]],
            sub_queries=["only one sub query"],  # one fewer than chains
            need_call_tools=True,
        )
        result = agent._trigger_workers(self._state(po, claried_query="clarified"), MagicMock())
        # Second chain falls back to claried_query
        assert result[1].arg["sub_query"] == "clarified"

    def test_send_worker_state_carries_additional_material(self, agent):
        po = PlanningOutput(
            tool_chains=[["segment", "entity_recognition"]],
            sub_queries=["identify player"],
            need_call_tools=True,
        )
        result = agent._trigger_workers(
            self._state(po, additional_material=["image.jpg"]),
            MagicMock(),
        )
        assert result[0].arg["additional_material"] == ["image.jpg"]

    def test_worker_state_initialised_with_empty_tool_histories(self, agent):
        po = PlanningOutput(
            tool_chains=[["game_search"]],
            sub_queries=["q"],
            need_call_tools=True,
        )
        result = agent._trigger_workers(self._state(po), MagicMock())
        ws = result[0].arg
        assert ws["tool_calls_history"] == []
        assert ws["tool_results_history"] == []
        assert ws["last_tool_artifact"] is None
        assert ws["messages"] == []


# ---------------------------------------------------------------------------
# _retrieve_cases_node
# ---------------------------------------------------------------------------

class TestRetrieveCasesNode:

    @pytest.mark.asyncio
    async def test_returns_cached_result_and_skips_retriever(self, agent):
        with patch("app.soccer_agent.agent.case_bank_cache") as mock_cache:
            mock_cache.get.return_value = "cached examples"
            state = {"claried_query": "Who won?", "additional_material": [], "user_query": "original"}
            result = await agent._retrieve_cases_node(state)
        assert result == {"retrieved_cases": "cached examples"}
        agent.case_bank_retriever.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_retriever_on_cache_miss_and_stores_result(self, agent):
        agent.case_bank_retriever.retrieve = AsyncMock(return_value="retrieved examples")
        with patch("app.soccer_agent.agent.case_bank_cache") as mock_cache:
            mock_cache.get.return_value = None
            state = {"claried_query": "Who won?", "additional_material": [], "user_query": "q"}
            result = await agent._retrieve_cases_node(state)
        assert result == {"retrieved_cases": "retrieved examples"}
        mock_cache.set.assert_called_once_with("Who won?", False, "retrieved examples")

    @pytest.mark.asyncio
    async def test_returns_none_when_retriever_returns_empty_string(self, agent):
        agent.case_bank_retriever.retrieve = AsyncMock(return_value="")
        with patch("app.soccer_agent.agent.case_bank_cache") as mock_cache:
            mock_cache.get.return_value = None
            state = {"claried_query": "q", "additional_material": [], "user_query": "q"}
            result = await agent._retrieve_cases_node(state)
        assert result == {"retrieved_cases": None}
        mock_cache.set.assert_not_called()  # empty result should not be cached

    @pytest.mark.asyncio
    async def test_falls_back_to_user_query_when_claried_query_empty(self, agent):
        agent.case_bank_retriever.retrieve = AsyncMock(return_value="")
        with patch("app.soccer_agent.agent.case_bank_cache") as mock_cache:
            mock_cache.get.return_value = None
            state = {
                "claried_query": "",
                "additional_material": [],
                "messages": [HumanMessage(content="original query")],
            }
            await agent._retrieve_cases_node(state)
        # Cache lookup should fall back to the last message content when claried_query is falsy
        mock_cache.get.assert_called_once_with("original query", False)

    @pytest.mark.asyncio
    async def test_has_media_true_when_additional_material_present(self, agent):
        agent.case_bank_retriever.retrieve = AsyncMock(return_value="")
        with patch("app.soccer_agent.agent.case_bank_cache") as mock_cache:
            mock_cache.get.return_value = None
            state = {"claried_query": "q", "additional_material": ["img.jpg"], "user_query": "q"}
            await agent._retrieve_cases_node(state)
        mock_cache.get.assert_called_once_with("q", True)  # has_media=True


# ---------------------------------------------------------------------------
# _aggregator_node
# ---------------------------------------------------------------------------

class TestAggregatorNode:

    def _make_state(self, parallel_results, claried_query="", messages_content="user Q"):
        msg = MagicMock()
        msg.content = messages_content
        return {
            "messages": [msg],
            "claried_query": claried_query,
            "additional_material": [],
            "conversation_history": "no history",
            "parallel_results": parallel_results,
        }

    @pytest.mark.asyncio
    async def test_returns_aggregator_response_as_message(self, agent):
        response = AIMessage(content="Final answer.")
        agent.aggregator_llm.ainvoke = AsyncMock(return_value=response)
        with patch("app.soccer_agent.agent.get_aggregator_prompt_template") as mock_tmpl:
            mock_tmpl.return_value.invoke.return_value = MagicMock()
            result = await agent._aggregator_node(self._make_state(["worker result"]), MagicMock())
        assert result["messages"] == [response]

    @pytest.mark.asyncio
    async def test_uses_claried_query_over_raw_user_query(self, agent):
        agent.aggregator_llm.ainvoke = AsyncMock(return_value=MagicMock())
        with patch("app.soccer_agent.agent.get_aggregator_prompt_template") as mock_tmpl:
            mock_tmpl.return_value.invoke.return_value = MagicMock()
            state = self._make_state(["r"], claried_query="clarified version", messages_content="original")
            await agent._aggregator_node(state, MagicMock())
        call_kwargs = mock_tmpl.return_value.invoke.call_args[0][0]
        assert call_kwargs["user_query"] == "clarified version"

    @pytest.mark.asyncio
    async def test_falls_back_to_message_content_when_claried_query_empty(self, agent):
        agent.aggregator_llm.ainvoke = AsyncMock(return_value=MagicMock())
        with patch("app.soccer_agent.agent.get_aggregator_prompt_template") as mock_tmpl:
            mock_tmpl.return_value.invoke.return_value = MagicMock()
            state = self._make_state(["r"], claried_query="", messages_content="raw user query")
            await agent._aggregator_node(state, MagicMock())
        call_kwargs = mock_tmpl.return_value.invoke.call_args[0][0]
        assert call_kwargs["user_query"] == "raw user query"

    @pytest.mark.asyncio
    async def test_no_tools_executed_message_when_parallel_results_empty(self, agent):
        agent.aggregator_llm.ainvoke = AsyncMock(return_value=MagicMock())
        with patch("app.soccer_agent.agent.get_aggregator_prompt_template") as mock_tmpl:
            mock_tmpl.return_value.invoke.return_value = MagicMock()
            state = self._make_state(parallel_results=[])
            await agent._aggregator_node(state, MagicMock())
        call_kwargs = mock_tmpl.return_value.invoke.call_args[0][0]
        assert "No tools were executed" in call_kwargs["worker_results"]

    @pytest.mark.asyncio
    async def test_multiple_results_numbered_in_worker_results_string(self, agent):
        agent.aggregator_llm.ainvoke = AsyncMock(return_value=MagicMock())
        with patch("app.soccer_agent.agent.get_aggregator_prompt_template") as mock_tmpl:
            mock_tmpl.return_value.invoke.return_value = MagicMock()
            state = self._make_state(parallel_results=["result A", "result B"])
            await agent._aggregator_node(state, MagicMock())
        call_kwargs = mock_tmpl.return_value.invoke.call_args[0][0]
        assert "Worker 1" in call_kwargs["worker_results"]
        assert "Worker 2" in call_kwargs["worker_results"]
        assert "result A" in call_kwargs["worker_results"]
        assert "result B" in call_kwargs["worker_results"]


# ---------------------------------------------------------------------------
# _worker_node
# ---------------------------------------------------------------------------

class TestWorkerNode:

    @pytest.mark.asyncio
    async def test_returns_parallel_results_from_worker_graph(self, agent):
        agent.worker_graph.ainvoke = AsyncMock(return_value={
            "worker_result": ["answer from worker"],
            "tool_calls_history": [{"name": "game_search", "args": {}}],
            "tool_results_history": [],
        })
        state = {"sub_query": "Who won?", "tool_chain": ["game_search"]}
        result = await agent._worker_node(state, MagicMock())
        assert result["parallel_results"] == ["answer from worker"]
        assert result["tool_calls_history"] == [{"name": "game_search", "args": {}}]
        assert result["tool_results_history"] == []

    @pytest.mark.asyncio
    async def test_returns_empty_results_when_worker_result_missing_in_output(self, agent):
        agent.worker_graph.ainvoke = AsyncMock(return_value={})
        state = {"sub_query": "q", "tool_chain": []}
        result = await agent._worker_node(state, MagicMock())
        assert result["parallel_results"] == []

    @pytest.mark.asyncio
    async def test_handles_timeout_and_returns_timeout_error_message(self, agent):
        agent.worker_graph.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())
        state = {"sub_query": "slow query", "tool_chain": ["heavy_tool"]}
        result = await agent._worker_node(state, MagicMock())
        assert len(result["parallel_results"]) == 1
        assert "[Timeout]" in result["parallel_results"][0]
        assert "slow query" in result["parallel_results"][0]
        assert result["tool_calls_history"] == []

    @pytest.mark.asyncio
    async def test_handles_generic_exception_and_returns_error_message(self, agent):
        agent.worker_graph.ainvoke = AsyncMock(side_effect=RuntimeError("DB down"))
        state = {"sub_query": "failing query", "tool_chain": ["game_search"]}
        result = await agent._worker_node(state, MagicMock())
        assert "[Error]" in result["parallel_results"][0]
        assert "DB down" in result["parallel_results"][0]
        assert result["tool_calls_history"] == []

    @pytest.mark.asyncio
    async def test_unknown_sub_query_shown_in_timeout_message(self, agent):
        agent.worker_graph.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())
        result = await agent._worker_node({}, MagicMock())  # no sub_query key
        assert "unknown" in result["parallel_results"][0]


# ---------------------------------------------------------------------------
# _execution_node
# ---------------------------------------------------------------------------

class TestExecutionNode:

    def _make_state(
        self,
        tool_chain=None,
        messages=None,
        last_tool_artifact=None,
        sub_query="Who won?",
    ):
        return {
            "sub_query": sub_query,
            "additional_material": [],
            "tool_chain": tool_chain or [],
            "tool_calls_history": [],
            "tool_results_history": [],
            "messages": messages or [],
            "last_tool_artifact": last_tool_artifact,
        }

    @pytest.mark.asyncio
    async def test_uses_tool_message_content_for_entity_augment(self, agent):
        """When last message is from entity_augment, its content wins."""
        response = AIMessage(content="LLM text", tool_calls=[])
        agent.execution_llm_with_tools.ainvoke = AsyncMock(return_value=response)

        tool_msg = ToolMessage(content="Retrieved from KB.", tool_call_id="t1", name="entity_augment")

        with patch("app.soccer_agent.agent.get_execution_system_prompt", return_value=AIMessage(content="sys")), \
             patch("app.soccer_agent.agent.semantic_cache"):
            result = await agent._execution_node(
                self._make_state(
                    tool_chain=["entity_augment"],
                    messages=[tool_msg],
                ),
                MagicMock(),
            )
        assert result["worker_result"] == ["Retrieved from KB."]

    @pytest.mark.asyncio
    async def test_uses_tool_message_content_for_game_history_retrieval(self, agent):
        response = AIMessage(content="LLM text", tool_calls=[])
        agent.execution_llm_with_tools.ainvoke = AsyncMock(return_value=response)

        tool_msg = ToolMessage(content="Match events.", tool_call_id="t1", name="game_history_retrieval")

        with patch("app.soccer_agent.agent.get_execution_system_prompt", return_value=AIMessage(content="sys")), \
             patch("app.soccer_agent.agent.semantic_cache"):
            result = await agent._execution_node(
                self._make_state(
                    tool_chain=["game_search", "game_history_retrieval"],
                    messages=[tool_msg],
                ),
                MagicMock(),
            )
        assert result["worker_result"] == ["Match events."]

    @pytest.mark.asyncio
    async def test_uses_tool_message_content_for_game_info_retrieval(self, agent):
        response = AIMessage(content="LLM text", tool_calls=[])
        agent.execution_llm_with_tools.ainvoke = AsyncMock(return_value=response)

        tool_msg = ToolMessage(content="Match metadata.", tool_call_id="t1", name="game_info_retrieval")

        with patch("app.soccer_agent.agent.get_execution_system_prompt", return_value=AIMessage(content="sys")), \
             patch("app.soccer_agent.agent.semantic_cache"):
            result = await agent._execution_node(
                self._make_state(messages=[tool_msg]),
                MagicMock(),
            )
        assert result["worker_result"] == ["Match metadata."]

    @pytest.mark.asyncio
    async def test_uses_llm_response_text_when_last_message_is_not_retrieval_tool(self, agent):
        """For non-retrieval tool results, use the LLM's text response."""
        response = AIMessage(content="Direct LLM answer.", tool_calls=[])
        agent.execution_llm_with_tools.ainvoke = AsyncMock(return_value=response)

        tool_msg = ToolMessage(content="Entity artifact.", tool_call_id="t1", name="choice_selection")

        with patch("app.soccer_agent.agent.get_execution_system_prompt", return_value=AIMessage(content="sys")), \
             patch("app.soccer_agent.agent.semantic_cache"):
            result = await agent._execution_node(
                self._make_state(messages=[tool_msg]),
                MagicMock(),
            )
        assert result["worker_result"] == ["Direct LLM answer."]

    @pytest.mark.asyncio
    async def test_uses_llm_response_when_no_prior_tool_messages(self, agent):
        """First turn in a chain: no ToolMessage in history → use LLM text."""
        response = AIMessage(content="LLM answer with no tools.", tool_calls=[])
        agent.execution_llm_with_tools.ainvoke = AsyncMock(return_value=response)

        with patch("app.soccer_agent.agent.get_execution_system_prompt", return_value=AIMessage(content="sys")), \
             patch("app.soccer_agent.agent.get_execution_human_prompt") as mock_human, \
             patch("app.soccer_agent.agent.semantic_cache"):
            mock_human.return_value.format.return_value = HumanMessage(content="human prompt")
            result = await agent._execution_node(
                self._make_state(messages=[]),  # no prior messages
                MagicMock(),
            )
        assert result["worker_result"] == ["LLM answer with no tools."]

    @pytest.mark.asyncio
    async def test_returns_empty_worker_result_when_llm_has_tool_calls(self, agent):
        """When LLM requests a tool call, we continue — no final result yet."""
        response = AIMessage(
            content="",
            tool_calls=[{"name": "game_search", "id": "call1", "args": {"query": "match"}}],
        )
        agent.execution_llm_with_tools.ainvoke = AsyncMock(return_value=response)

        with patch("app.soccer_agent.agent.get_execution_system_prompt", return_value=AIMessage(content="sys")), \
             patch("app.soccer_agent.agent.get_execution_human_prompt") as mock_human, \
             patch("app.soccer_agent.agent.semantic_cache"):
            mock_human.return_value.format.return_value = HumanMessage(content="prompt")
            result = await agent._execution_node(self._make_state(), MagicMock())
        # No final result stored when tool call is pending
        assert result["worker_result"] == []

    @pytest.mark.asyncio
    async def test_stores_result_in_semantic_cache_on_completion(self, agent):
        response = AIMessage(content="Cached answer.", tool_calls=[])
        agent.execution_llm_with_tools.ainvoke = AsyncMock(return_value=response)

        with patch("app.soccer_agent.agent.get_execution_system_prompt", return_value=AIMessage(content="sys")), \
             patch("app.soccer_agent.agent.get_execution_human_prompt") as mock_human, \
             patch("app.soccer_agent.agent.semantic_cache") as mock_cache:
            mock_human.return_value.format.return_value = HumanMessage(content="prompt")
            await agent._execution_node(self._make_state(sub_query="test query"), MagicMock())
        mock_cache.set.assert_called_once_with("test query", "Cached answer.", [])

    @pytest.mark.asyncio
    async def test_returns_error_worker_result_when_llm_raises(self, agent):
        agent.execution_llm_with_tools.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

        with patch("app.soccer_agent.agent.get_execution_system_prompt", return_value=AIMessage(content="sys")), \
             patch("app.soccer_agent.agent.get_execution_human_prompt") as mock_human, \
             patch("app.soccer_agent.agent.semantic_cache"):
            mock_human.return_value.format.return_value = HumanMessage(content="prompt")
            result = await agent._execution_node(self._make_state(), MagicMock())
        # After error, LLM returns a fallback AIMessage with no tool calls.
        # The response.text is the agent's error message (not the "Worker stopped" branch,
        # because response.text is truthy after the error fallback is constructed).
        assert "execution has been stopped" in result["worker_result"][0]

    @pytest.mark.asyncio
    async def test_artifact_from_last_tool_message_stored_in_state(self, agent):
        response = AIMessage(content="Answer.", tool_calls=[])
        agent.execution_llm_with_tools.ainvoke = AsyncMock(return_value=response)

        tool_msg = ToolMessage(
            content="Match info.",
            tool_call_id="t1",
            name="game_info_retrieval",
            artifact="england_epl/game123",
        )
        with patch("app.soccer_agent.agent.get_execution_system_prompt", return_value=AIMessage(content="sys")), \
             patch("app.soccer_agent.agent.semantic_cache"):
            result = await agent._execution_node(
                self._make_state(messages=[tool_msg]),
                MagicMock(),
            )
        assert result["last_tool_artifact"] == "england_epl/game123"
