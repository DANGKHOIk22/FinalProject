import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from app.config.config import GUARDRAIL_REFUSAL_MESSAGE
from app.schema.soccer_agent.state import GuardrailVerdict
from app.soccer_agent.nodes.guardrail import GuardrailNode


def _make_node(verdict: GuardrailVerdict | None = None, ainvoke_side_effect=None):
    """Build a GuardrailNode whose structured classifier is fully mocked."""
    llm = MagicMock()
    structured = MagicMock()
    if ainvoke_side_effect is not None:
        structured.ainvoke = AsyncMock(side_effect=ainvoke_side_effect)
    else:
        structured.ainvoke = AsyncMock(return_value=verdict)
    llm.with_structured_output.return_value = structured
    return GuardrailNode(llm), structured


# --- classify_node: happy paths -------------------------------------------------

@pytest.mark.asyncio
async def test_off_topic_query_marked_off_topic():
    node, _ = _make_node(GuardrailVerdict(is_soccer_related=False, reason="cooking"))
    state = {"messages": [HumanMessage(content="cho tôi công thức nấu phở")]}
    result = await node.classify_node(state, {})
    assert result["is_off_topic"] is True


@pytest.mark.asyncio
async def test_soccer_query_not_off_topic():
    node, _ = _make_node(GuardrailVerdict(is_soccer_related=True, reason="player"))
    state = {"messages": [HumanMessage(content="Who is Lionel Messi?")]}
    result = await node.classify_node(state, {})
    assert result["is_off_topic"] is False


# --- classify_node: context awareness (R4) -------------------------------------

@pytest.mark.asyncio
async def test_recent_context_passed_to_classifier():
    node, structured = _make_node(GuardrailVerdict(is_soccer_related=True, reason="follow-up"))
    state = {
        "messages": [
            HumanMessage(content="Tell me about Messi"),
            AIMessage(content="Messi plays for Inter Miami."),
            HumanMessage(content="and his goals this season?"),
        ]
    }
    await node.classify_node(state, {})
    prompt_messages = structured.ainvoke.await_args.args[0]
    joined = " ".join(getattr(m, "content", "") for m in prompt_messages)
    assert "Messi" in joined  # prior turn surfaced as context
    assert "and his goals this season?" in joined  # current query present


# --- classify_node: allow-bias and edge cases (R5) -----------------------------

@pytest.mark.asyncio
async def test_ambiguous_allowed_when_model_returns_true():
    node, _ = _make_node(GuardrailVerdict(is_soccer_related=True, reason="ambiguous-allow"))
    state = {"messages": [HumanMessage(content="what happened yesterday?")]}
    result = await node.classify_node(state, {})
    assert result["is_off_topic"] is False


@pytest.mark.asyncio
async def test_empty_messages_fail_open_without_calling_llm():
    node, structured = _make_node(GuardrailVerdict(is_soccer_related=False, reason="x"))
    result = await node.classify_node({"messages": []}, {})
    assert result["is_off_topic"] is False
    structured.ainvoke.assert_not_awaited()


# --- classify_node: fail-open (R11) --------------------------------------------

@pytest.mark.asyncio
async def test_llm_error_fail_open():
    node, _ = _make_node(ainvoke_side_effect=RuntimeError("boom"))
    state = {"messages": [HumanMessage(content="give me a pasta recipe")]}
    result = await node.classify_node(state, {})
    assert result["is_off_topic"] is False


@pytest.mark.asyncio
async def test_timeout_fail_open():
    node, _ = _make_node(ainvoke_side_effect=asyncio.TimeoutError())
    state = {"messages": [HumanMessage(content="give me a pasta recipe")]}
    result = await node.classify_node(state, {})
    assert result["is_off_topic"] is False


# --- gate_router ---------------------------------------------------------------

def test_gate_router_blocked():
    node, _ = _make_node(GuardrailVerdict(is_soccer_related=False, reason="x"))
    assert node.gate_router({"is_off_topic": True}) == "blocked"


def test_gate_router_proceed_on_false_or_missing():
    node, _ = _make_node(GuardrailVerdict(is_soccer_related=True, reason="x"))
    assert node.gate_router({"is_off_topic": False}) == "proceed"
    assert node.gate_router({}) == "proceed"


# --- gate_node (no-op join) ----------------------------------------------------

def test_gate_node_is_noop():
    node, _ = _make_node(GuardrailVerdict(is_soccer_related=True, reason="x"))
    assert node.gate_node({"is_off_topic": True}) == {}


# --- refusal_node --------------------------------------------------------------

@pytest.mark.asyncio
async def test_refusal_node_emits_fixed_message():
    node, _ = _make_node(GuardrailVerdict(is_soccer_related=False, reason="x"))
    result = await node.refusal_node({"messages": [HumanMessage(content="cooking")]}, {})
    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)
    assert msgs[0].content == GUARDRAIL_REFUSAL_MESSAGE
