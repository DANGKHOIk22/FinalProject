import logging
import uuid
import asyncio
from datetime import date
from typing import List

from langfuse import get_client
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph.state import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser

from app.schema.soccer_agent.state import AgentState, PlanningOutput
from app.soccer_agent.prompts.agent import get_planning_prompt_template

logger = logging.getLogger(__name__)

class ToolChainPlanningNode:
    def __init__(self, planning_llm, planning_parser: PydanticOutputParser, tools: List[BaseTool]):
        self.planning_llm = planning_llm
        self.planning_parser = planning_parser
        self.tools = tools

    async def tool_chain_planning(self, state: AgentState, config: RunnableConfig) -> dict:
        """
        This node is responsible for planning the tool chain to answer the user's query.
        """
        messages = state.get("messages", [])
        user_query = messages[-1].content if messages else ""
        additional_material_list = state.get("additional_material")
        additional_material = ", ".join(additional_material_list) if additional_material_list else "None"
        conversation_history = state.get("conversation_history", "No previous conversation.")

        # --- Flags for tracing metadata ---
        need_call_tool = True # True/False: Indicates whether tool calls are necessary

        # --- Emit a tool call event to show the planning step in the UI ---
        await adispatch_custom_event(
            "manually_emit_tool_call",
            data={
                "id": str(uuid.uuid4()),
                "name": "tool_chain_planning",
                "args": {}
            },
            config=config
        )
        await asyncio.sleep(0.1) 
    
        logger.info("="*70)
        logger.info("🧠 Starting TOOL CHAIN PLANNING STEP")
        
        # Create prompt for planning agent
        tool_descriptions = ""
        for tool in self.tools:
            tool_descriptions += f"- {tool.name}: {tool.description}\n"
        
        format_instructions = self.planning_parser.get_format_instructions()
        planning_agent_prompt_template = get_planning_prompt_template()
        
        planning_agent_prompt = planning_agent_prompt_template.invoke({
            "toolbox_descriptions": tool_descriptions,
            "format_instructions": format_instructions,
            "user_query": state.get("claried_query") or user_query,
            "additional_material": additional_material,
            "conversation_history": conversation_history,
            "retrieved_cases": state.get("retrieved_cases") or "",
            "current_date": date.today().isoformat(),
        })
            
        # Modify the config metadata to skip emit planning agent response
        config.update(metadata={
            "copilotkit:emit-messages": False,
            "copilotkit:emit-tool_calls": False,
        })
        response = await self.planning_llm.ainvoke(planning_agent_prompt, config=config)
        response_text = response.text if hasattr(response, 'text') else str(response)
            
        planning_output: PlanningOutput = self.planning_parser.parse(response_text)
        need_call_tool = planning_output.need_call_tools
        
        logger.info("Tool Chain Planning Results:")
        logger.info(f"\t Tool Chains: {planning_output.tool_chains}")
        logger.info("✅ TOOL CHAIN PLANNING STEP COMPLETED")
        logger.info("="*70)

        # --- Update tracing metadata ---
        langfuse = get_client()
        langfuse.update_current_span(
            metadata={
                "need_call_tool": need_call_tool
            }
        )
        return {
            "additional_material": sorted(state.get("additional_material") or []),
            "planning_output": planning_output,
            "conversation_history": state.get("conversation_history", ""),
        }
