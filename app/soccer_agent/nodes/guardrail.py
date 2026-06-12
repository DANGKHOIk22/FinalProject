import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import RunnableConfig

from app.config.config import (
    GUARDRAIL_RECENT_TURNS,
    GUARDRAIL_REFUSAL_MESSAGE,
    GUARDRAIL_TIMEOUT_SECONDS,
)
from app.schema.soccer_agent.state import AgentState, GuardrailVerdict
from app.soccer_agent.prompts.agent import get_guardrail_prompt_template

logger = logging.getLogger(__name__)


class GuardrailNode:
    """Soccer-topic guardrail.

    Runs in parallel with get_history / context_retrieval. `classify_node` writes
    `is_off_topic` to state; `gate_node` joins the three parallel branches and
    `gate_router` routes off-topic queries to `refusal_node` (a canned refusal),
    skipping planning / workers / aggregator entirely. Fails open: any classifier
    error or timeout lets the query proceed to planning.
    """

    def __init__(self, guardrail_llm):
        self.classifier = guardrail_llm.with_structured_output(GuardrailVerdict)

    @staticmethod
    def _current_query(messages) -> str | None:
        """The latest user query — last message if it's a HumanMessage, else the
        most recent HumanMessage. Returns None when there is nothing to classify."""
        if not messages:
            return None
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return last.text
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.text
        return None

    @staticmethod
    def _recent_context(messages) -> str:
        """Format the trailing turns (excluding the current query) as context."""
        prior = messages[:-1][-GUARDRAIL_RECENT_TURNS:]
        if not prior:
            return "(none)"
        lines = []
        for msg in prior:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            if msg.text:
                lines.append(f"{role}: {msg.text}")
        return "\n".join(lines) if lines else "(none)"

    async def classify_node(self, state: AgentState, config: RunnableConfig) -> dict:
        """Classify the current query as soccer-related or not. Fail-open."""
        messages = state.get("messages", [])
        query = self._current_query(messages)
        if not query:
            logger.info("[Guardrail] No query to classify — allowing through.")
            return {"is_off_topic": False}

        prompt_value = get_guardrail_prompt_template().invoke({
            "recent_context": self._recent_context(messages),
            "current_query": query,
        })

        try:
            verdict: GuardrailVerdict = await asyncio.wait_for(
                self.classifier.ainvoke(prompt_value.to_messages(), config=config),
                timeout=GUARDRAIL_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning(f"[Guardrail] Classification failed ({e!r}) — failing open (allow).")
            return {"is_off_topic": False}

        is_off_topic = not verdict.is_soccer_related
        logger.info(
            f"[Guardrail] off_topic={is_off_topic} (reason: {verdict.reason}) "
            f"for query: '{query[:60]}'"
        )
        return {"is_off_topic": is_off_topic}

    def gate_node(self, state: AgentState) -> dict:
        """No-op join: exists so get_history, context_retrieval, and the guardrail
        converge on a single node that owns the routing conditional."""
        return {}

    def gate_router(self, state: AgentState) -> str:
        """Route to the refusal node when off-topic, else to planning."""
        return "blocked" if state.get("is_off_topic") else "proceed"

    async def refusal_node(self, state: AgentState, config: RunnableConfig) -> dict:
        """Emit the canned refusal. The current-turn HumanMessage is already in
        `messages` (seeded by run() / supplied by the streaming adapter), so
        save_to_memory_node's dual-message guard fires and the turn is persisted."""
        logger.info("[Guardrail] Off-topic query blocked — returning canned refusal.")
        return {"messages": [AIMessage(content=GUARDRAIL_REFUSAL_MESSAGE)]}
