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

    @staticmethod
    def extract_media_uuids(messages: list, config: RunnableConfig, additional_material: dict) -> dict:
        """
        Extract additional material UUIDs from messages if media registry is available
        and update the additional_material dictionary.
        """
        media_registry = config.get("configurable", {}).get("media_registry")
        if media_registry and messages:
            extracted_media = media_registry.extract_uuids_from_message(messages[-1])
            
            # Process image_id list
            extracted_images = extracted_media.get("image_ids") or []
            if extracted_images:
                additional_material["image_id"] = extracted_images
                logger.info(f"Extracted SAS URL image UUIDs from message: {extracted_images}")
            else:
                additional_material["image_id"] = []

            # Process video_id (we support single video upload)
            extracted_video = extracted_media.get("video_id")
            if extracted_video:
                additional_material["video_id"] = extracted_video
                logger.info(f"Extracted SAS URL video UUID from message: {extracted_video}")
            else:
                additional_material["video_id"] = None
        else:
            additional_material["image_id"] = []
            additional_material["video_id"] = None
        return additional_material

    @staticmethod
    def extract_game_context(state: AgentState, additional_material: dict) -> tuple:
        """
        Extract game_id and video_current_time from additional_material/state/copilotkit context.
        Also persists game_id in additional_material.
        """
        video_current_time = state.get("video_current_time")
        game_id = additional_material.get("game_id")
        # Always scan CopilotKit context for the latest current_time — even when game_id is
        # already known from a prior checkpoint. Without this, video_current_time freezes at
        # the value persisted during the first turn and never updates across turns.
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
                if game_id is None:
                    game_id = value["game_id"]
                video_current_time = value.get("current_time", video_current_time)
                break
        # Persist the resolved video context so downstream nodes (trigger_workers,
        # cache, tools) read it from state instead of re-parsing the context blob.
        additional_material["game_id"] = game_id
        return game_id, video_current_time

    @staticmethod
    def remove_video_content(message: HumanMessage) -> HumanMessage:
        """
        Remove content parts of type 'video_url' from the HumanMessage content
        so that the video block is not seen by the planning node.
        """
        if not message or not hasattr(message, "content"):
            return message

        content = message.content
        if isinstance(content, list):
            new_content = [
                part for part in content
                if not (isinstance(part, dict) and part.get("type") == "video_url")
            ]
            import copy
            try:
                return message.model_copy(update={"content": new_content})
            except Exception:
                try:
                    return message.copy(update={"content": new_content})
                except Exception:
                    try:
                        new_msg = copy.copy(message)
                        new_msg.content = new_content
                        return new_msg
                    except Exception:
                        return message
        return message

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
        conversation_history = state.get("conversation_history") or "No previous conversation."
        retrieved_cases = state.get("retrieved_cases") or "No examples available."

        # Extract additional material UUIDs from messages if media registry is available
        additional_material = self.extract_media_uuids(messages, config, additional_material)

        # Extract game_id and video_current_time from additional_material for prompt context
        game_id, video_current_time = self.extract_game_context(state, additional_material)

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

        if additional_material.get("image_id") or additional_material.get("video_id"):
            game_id = None
            
        prompt_template = get_unified_planning_prompt_template()
        prompt_value = prompt_template.invoke({
            "user_query_msg": [self.remove_video_content(messages[-1])] if messages else [],
            "image_ids": ", ".join(additional_material.get("image_id") or []) or "None",
            "video_id": additional_material.get("video_id") or "None",
            "conversation_history": conversation_history,
            "toolbox_descriptions": toolbox_descriptions,
            "retrieved_cases": retrieved_cases,
            "long_term_context": state.get("long_term_context") or "No relevant long-term memory found.",
            "time_context": state.get("time_context") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "game_id": game_id or "No game context",
            "video_current_time": video_current_time if video_current_time is not None else "No video context",
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
