import json
import logging
import uuid
from datetime import datetime
from app.config.config import PLANNING_CONFIDENCE_THRESHOLD
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph.state import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage

from app.schema.soccer_agent.state import AgentState, UnifiedPlanningOutput
from app.soccer_agent.prompts.agent import get_unified_planning_prompt_template

logger = logging.getLogger(__name__)

class UnifiedPlanningNode:
    def __init__(self, planning_llm, tools):
        self.planning_llm = planning_llm
        self.tools = tools
        self.parser = PydanticOutputParser(pydantic_object=UnifiedPlanningOutput)

    async def unified_planning_node(self, state: AgentState, config: RunnableConfig):
        """
        Combined node for Query Understanding and Tool Chain Planning.
        Produces per-chain confidence scores and splits chains into
        dispatchable (non-ambiguous) vs pending_clarifications (ambiguous).
        """
        messages = state.get("messages", [])
        user_query = str(messages[-1].text) if messages else ""
        # Copy — never mutate the dict held by the graph state
        additional_material = dict(state.get("additional_material") or {})

        # Extract additional material UUIDs from messages if media registry is available
        media_registry = config.get("configurable", {}).get("media_registry")
        if media_registry and messages:
            extracted_uuids = media_registry.extract_uuids_from_message(messages[-1])
            if extracted_uuids:
                existing = additional_material.get("image_id") or []
                new_ids = [u for u in extracted_uuids if u not in existing]
                if new_ids:
                    additional_material["image_id"] = existing + new_ids
                    logger.info(f"Extracted SAS URL UUIDs from message: {new_ids}")

        conversation_history = state.get("conversation_history") or "No previous conversation."
        retrieved_cases = state.get("retrieved_cases") or "No examples available."

        toolbox_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])
        try:
            await adispatch_custom_event(
                "manually_emit_tool_call",
                data={
                    "id": str(uuid.uuid4()),
                    "name": "tool_chain_planning",
                    "args": {"query": user_query[:50] + "..."}
                },
                config=config
            )
        except RuntimeError as e:
            logger.warning(
                f"Failed to dispatch custom event: {e}. "
                "This is expected if running outside a LangChain/LangGraph run context (e.g., in unit tests)."
            )

        video_current_time = state.get("video_current_time")
        game_id = additional_material.get("game_id")
        if game_id is None:
            for ctx_item in state.get("copilotkit", {}).get("context", []):
                raw = ctx_item.value if hasattr(ctx_item, "value") else ctx_item.get("value")
                if isinstance(raw, dict):
                    value = raw
                elif isinstance(raw, str):
                    # Frontend-controlled payload — never trust it to be valid JSON
                    try:
                        value = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(f"Skipping malformed CopilotKit context value: {raw[:100]!r}")
                        value = None
                else:
                    value = None
                if isinstance(value, dict) and value.get("game_id"):
                    game_id = value["game_id"]
                    video_current_time = value.get("current_time", video_current_time)
                    break
        # Persist the resolved video context so downstream nodes (trigger_workers,
        # cache, tools) read it from state instead of re-parsing the context blob.
        additional_material["game_id"] = game_id
        video_context = (
            f"HLS game_id={game_id}, current_time={video_current_time}s"
            if game_id is not None
            else "None"
        )
        
        prompt_template = get_unified_planning_prompt_template()
        prompt_value = prompt_template.invoke({
            "user_query_msg": [messages[-1]] if messages else [],
            "additional_material": ", ".join(additional_material.get("image_id") or []) or "None",
            "conversation_history": conversation_history,
            "toolbox_descriptions": toolbox_descriptions,
            "retrieved_cases": retrieved_cases,
            "long_term_context": state.get("long_term_context") or "No relevant long-term memory found.",
            "time_context": state.get("time_context") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_context": video_context,
            "format_instructions": self.parser.get_format_instructions()
        })
        prompt_messages = prompt_value.to_messages()

        response = await self.planning_llm.ainvoke(prompt_messages, config=config)
        response_text = response.text if hasattr(response, 'text') else str(response.content)
    
        try:
            output: UnifiedPlanningOutput = self.parser.parse(response_text)
        except Exception as e:
            logger.error(f"Failed to parse UnifiedPlanningOutput: {e}")
            # Mark the failure explicitly — the aggregator must surface a system
            # error, not disguise it as "your question is unclear".
            return {
                "clarified_query": user_query,
                "planning_output": None,
                "tool_chains": [],
                "sub_queries": [],
                "need_call_tools": False,
                "pending_clarifications": [],
                "planning_error": str(e),
                "additional_material": additional_material,
                "video_current_time": video_current_time,
            }

        # Split planned_chains into dispatchable vs ambiguous
        tool_chains: list = []
        sub_queries: list = []
        pending_clarifications: list = []

        for pc in (output.planned_chains or []):
            # Low-confidence chains are treated as ambiguous.
            if pc.confidence < PLANNING_CONFIDENCE_THRESHOLD:
                pc.is_ambiguous = True
            # Clarification gate (scoped): only ask the user when the chain is ambiguous
            # AND there is no active video to anchor it to. When a game_id is present the
            # chain is dispatched and resolved against the watched match instead of asking.
            if pc.is_ambiguous and game_id is None:
                q = pc.clarifying_question or "Bạn có thể cung cấp thêm thông tin không?"
                pending_clarifications.append(q)
                logger.info(f"  └─ Chain AMBIGUOUS (confidence={pc.confidence:.2f}): {pc.chain} | '{pc.sub_query}' → asking: {q}")
            else:
                tool_chains.append(pc.chain)
                sub_queries.append(pc.sub_query)
                logger.info(f"  └─ Chain OK (confidence={pc.confidence:.2f}): {pc.chain} | '{pc.sub_query}'")

        need_call_tools = output.need_call_tools and len(tool_chains) > 0
        logger.info(
            f"[UnifiedPlanning] clarified='{output.clarified_query}' "
            f"dispatchable={len(tool_chains)} ambiguous={len(pending_clarifications)} "
            f"need_call_tools={need_call_tools}"
        )

        return {
            "clarified_query": output.clarified_query,
            "planning_output": output,
            "tool_chains": tool_chains,
            "sub_queries": sub_queries,
            "need_call_tools": need_call_tools,
            "pending_clarifications": pending_clarifications,
            "planning_error": None,  # clear any checkpointed error from a previous turn
            "additional_material": additional_material,
            "video_current_time": video_current_time,
        }
