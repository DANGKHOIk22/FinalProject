import logging
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph.state import RunnableConfig
from langgraph.types import Command
from langgraph.graph import END

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
        Reduces latency by saving one LLM round-trip.
        """
        messages = state.get("messages", [])
        user_query = state.get("user_query") or (messages[-1].content if messages else "")
        additional_material = state.get("additional_material", [])
        conversation_history = state.get("conversation_history") or "No previous conversation."
        retrieved_cases = state.get("retrieved_cases") or "No examples available."
        
        # Format tool descriptions
        toolbox_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])

        # Emit event for UI
        await adispatch_custom_event(
            "manually_emit_tool_call",
            data={
                "id": str(uuid.uuid4()),
                "name": "unified_planning",
                "args": {"query": user_query[:50] + "..."}
            },
            config=config
        )

        prompt_template = get_unified_planning_prompt_template()
        prompt = prompt_template.invoke({
            "user_query": user_query,
            "additional_material": ", ".join(additional_material) if additional_material else "None",
            "conversation_history": conversation_history,
            "toolbox_descriptions": toolbox_descriptions,
            "retrieved_cases": retrieved_cases,
            "long_term_context": state.get("long_term_context") or "No relevant long-term memory found.",
            "time_context": state.get("time_context") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "format_instructions": self.parser.get_format_instructions()
        })

        # Call LLM
        response = await self.planning_llm.ainvoke(prompt, config=config)
        
        # Use .text property if available (user preference in recent diffs)
        response_text = response.text if hasattr(response, 'text') else str(response.content)
        
        try:
            output: UnifiedPlanningOutput = self.parser.parse(response_text)
        except Exception as e:
            logger.error(f"Failed to parse UnifiedPlanningOutput: {e}")
            return {
                "clarified_query": user_query,
                "planning_output": None,
                "tool_chains": [],
                "sub_queries": [],
                "need_call_tools": False
            }

        logger.info(f"[UnifiedPlanning] clarified='{output.clarified_query}' is_ambiguous={output.is_ambiguous} tools={output.need_call_tools}")
        if output.need_call_tools and output.tool_chains:
            for idx, chain in enumerate(output.tool_chains):
                sub_q = output.sub_queries[idx] if idx < len(output.sub_queries) else "N/A"
                logger.info(f"  └─ Chain {idx}: {chain} | Sub-query: '{sub_q}'")

        # Handle Ambiguity Short-circuit
        if output.is_ambiguous and output.clarifying_questions and not additional_material:
            questions_text = "\n".join(f"- {q}" for q in output.clarifying_questions)
            clarification_response = AIMessage(content=(
                f"Câu hỏi của bạn chưa đủ rõ ràng để tôi trả lời chính xác. "
                f"Bạn có thể làm rõ thêm không?\n{questions_text}"
            ))
            return Command(
                goto=END,
                update={
                    "messages": messages + [clarification_response],
                },
            )

        return {
            "clarified_query": output.clarified_query,
            "planning_output": output, # Used by trigger_workers
            "tool_chains": output.tool_chains or [],
            "sub_queries": output.sub_queries or [],
            "need_call_tools": output.need_call_tools
        }
