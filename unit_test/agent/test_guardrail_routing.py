"""Integration test for guardrail routing in the main graph.

Builds a graph wired exactly like SoccerAgent._build_graph (guardrail subgraph
portion) using the REAL GuardrailNode, with lightweight stubs for the heavy
nodes. Constructing the full SoccerAgent needs API keys / DB, so we exercise the
routing contract directly: off-topic must skip planning and emit the refusal;
on-topic must reach planning.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from app.config.config import GUARDRAIL_REFUSAL_MESSAGE
from app.schema.soccer_agent.state import AgentState, GuardrailVerdict
from app.soccer_agent.nodes.guardrail import GuardrailNode


def _make_guardrail(is_soccer_related: bool) -> GuardrailNode:
    llm = MagicMock()
    structured = MagicMock()
    structured.ainvoke = AsyncMock(
        return_value=GuardrailVerdict(is_soccer_related=is_soccer_related, reason="test")
    )
    llm.with_structured_output.return_value = structured
    return GuardrailNode(llm)


def _build_graph(guardrail_node: GuardrailNode, planning_spy):
    wf = StateGraph(AgentState)

    async def get_history(state, config=None):
        return {}

    async def context_retrieval(state, config=None):
        return {}

    async def unified_planning(state, config=None):
        planning_spy()
        return {"messages": [AIMessage(content="planned response")]}

    wf.add_node("get_history", get_history)
    wf.add_node("context_retrieval", context_retrieval)
    wf.add_node("guardrail_classify", guardrail_node.classify_node)
    wf.add_node("guardrail_gate", guardrail_node.gate_node)
    wf.add_node("guardrail_refusal", guardrail_node.refusal_node)
    wf.add_node("unified_planning", unified_planning)

    wf.add_edge(START, "get_history")
    wf.add_edge(START, "context_retrieval")
    wf.add_edge(START, "guardrail_classify")
    wf.add_edge("get_history", "guardrail_gate")
    wf.add_edge("context_retrieval", "guardrail_gate")
    wf.add_edge("guardrail_classify", "guardrail_gate")
    wf.add_conditional_edges(
        "guardrail_gate",
        guardrail_node.gate_router,
        {"proceed": "unified_planning", "blocked": "guardrail_refusal"},
    )
    wf.add_edge("guardrail_refusal", END)
    wf.add_edge("unified_planning", END)
    return wf.compile()


@pytest.mark.asyncio
async def test_off_topic_skips_planning_and_returns_refusal():
    planning_spy = MagicMock()
    graph = _build_graph(_make_guardrail(is_soccer_related=False), planning_spy)

    final = await graph.ainvoke({"messages": [HumanMessage(content="give me a pasta recipe")]})

    assert final["messages"][-1].content == GUARDRAIL_REFUSAL_MESSAGE
    planning_spy.assert_not_called()  # planning is skipped entirely


@pytest.mark.asyncio
async def test_on_topic_proceeds_to_planning():
    planning_spy = MagicMock()
    graph = _build_graph(_make_guardrail(is_soccer_related=True), planning_spy)

    final = await graph.ainvoke({"messages": [HumanMessage(content="Who won the 2022 World Cup?")]})

    planning_spy.assert_called_once()
    assert final["messages"][-1].content == "planned response"


@pytest.mark.asyncio
async def test_off_topic_run_leaves_human_and_ai_messages_for_persistence():
    """save_to_memory_node needs both a HumanMessage and an AIMessage; confirm the
    refusal path leaves both so the turn persists (R8)."""
    planning_spy = MagicMock()
    graph = _build_graph(_make_guardrail(is_soccer_related=False), planning_spy)

    final = await graph.ainvoke({"messages": [HumanMessage(content="how do I bake bread?")]})

    msgs = final["messages"]
    assert any(isinstance(m, HumanMessage) for m in msgs)
    assert any(isinstance(m, AIMessage) and m.content == GUARDRAIL_REFUSAL_MESSAGE for m in msgs)
