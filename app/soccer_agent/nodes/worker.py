import logging
import asyncio
from datetime import datetime
from typing import List

from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from langgraph.types import Send
from langgraph.prebuilt import ToolNode

from app.schema.soccer_agent.state import AgentState, WorkerState
from app.soccer_agent.prompts.agent import get_execution_human_prompt, get_execution_system_prompt
from app.cache.semantic_cache import semantic_cache

logger = logging.getLogger(__name__)

class WorkerNodes:
    def __init__(self, execution_llm_with_tools, tools: List[BaseTool]):
        self.execution_llm_with_tools = execution_llm_with_tools
        self.tools = tools
        self.worker_graph = self._build_worker_graph()

    def _build_worker_graph(self) -> CompiledStateGraph:
        """Build the LangGraph worker workflow."""
        workflow = StateGraph(WorkerState)
        tool_node = ToolNode(self.tools)
        
        workflow.add_node("check_cache_node", self._check_cache_node)
        workflow.add_node("execution_node", self._execution_node)
        workflow.add_node("tool_node", tool_node)
        
        workflow.set_entry_point("check_cache_node")
        
        workflow.add_conditional_edges(
            "check_cache_node",
            self.should_execute_worker,
            {
                "execute": "execution_node",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "execution_node",
            self.should_continue_call_tool,
            {
                "continue": "tool_node",
                "end": END
            }
        )
        workflow.add_edge("tool_node", "execution_node")
        return workflow.compile()

    def trigger_workers(self, state: AgentState, config: RunnableConfig):
        """Map worker executions for each parallel tool chain."""
        # Read from top-level state — always freshly written by unified_planning_node.
        # Do NOT read from planning_output: it may be stale (from a previous turn's checkpointed state).
        need_call_tools = state.get("need_call_tools", True)
        tool_chains = state.get("tool_chains") or []
        sub_queries = state.get("sub_queries") or []
        messages = state.get("messages", [])
        clarified_query = state.get("clarified_query") or (messages[-1].content if messages else "")

        # Short-circuit to aggregator when no tools needed (is_ambiguous, greeting, etc.)
        if not need_call_tools or not tool_chains:
            logger.info(f"⏭️ Skipping workers (need_call_tools={need_call_tools}, tool_chains={tool_chains}). Going straight to aggregator.")
            return "aggregator"
            
        sends = []
        for idx, chain in enumerate(tool_chains):
            sub_query = sub_queries[idx] if idx < len(sub_queries) else clarified_query
            worker_state = {
                "messages": [], # Start with empty messages for the worker;
                "sub_query": sub_query,
                "additional_material": state.get("additional_material", []),
                "tool_chain": chain,
                "tool_calls_history": [],
                "tool_results_history": [],
                "last_tool_artifact": None,
                "parallel_results": [],
                "time_context": state.get("time_context") or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
            logger.info(f"🚀 Triggering worker {idx}: sub_query='{sub_query}', chain={chain}")
            sends.append(Send("worker_graph", worker_state))
        return sends

    async def worker_node(self, state: WorkerState, config: RunnableConfig) -> dict:
        """
        Wrapper for worker_graph with proper timeout and Langfuse tracing.
        """
        try:
            async def worker_execution():
                return await asyncio.wait_for(
                    self.worker_graph.ainvoke(state, config=config),
                    timeout=120.0
                )
        
            result = await worker_execution()
            
            return {
                "parallel_results": result.get("worker_result", []),
                "tool_calls_history": result.get("tool_calls_history", []),
                "tool_results_history": result.get("tool_results_history", []),
            }
        except asyncio.TimeoutError:
            error_msg = f"[Timeout] Worker for sub-query '{state.get('sub_query', 'unknown')}' exceeded 120 seconds."
            logger.error(error_msg)
            return {
                "parallel_results": [error_msg],
                "tool_calls_history": [],
                "tool_results_history": []
            }
        except Exception as e:
            error_msg = f"[Error] Worker for sub-query '{state.get('sub_query', 'unknown')}' failed with error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "parallel_results": [error_msg],
                "tool_calls_history": [],
                "tool_results_history": [],
            }

    async def _execution_node(self, state: WorkerState, config: RunnableConfig) -> dict:
        """
        Iteratively execute the tool chain step by step using bind_tools with tool_choice.
        """
        sub_query = state.get("sub_query")
        additional_material_list = state.get("additional_material", [])
        tool_chain = state["tool_chain"]
        tool_calls_history = state.get("tool_calls_history", [])
        tool_results_history = state.get("tool_results_history", [])
        messages = state.get("messages", [])

        logger.info(f"🔧 Running TOOL EXECUTION STEP: Step {len(tool_calls_history)}")
        
        last_artifact = state.get("last_tool_artifact")
        last_tool_message = None
        if messages and isinstance(messages[-1], ToolMessage):
            last_tool_message = messages[-1]
            last_artifact = last_tool_message.artifact if hasattr(last_tool_message, 'artifact') else None

        additional_material_str = ", ".join(additional_material_list) if additional_material_list else "None"
        system_prompt = get_execution_system_prompt()
        if not messages:
            execution_prompt_template = get_execution_human_prompt()
            execution_prompt = execution_prompt_template.format(
                sub_query=sub_query,
                additional_material=additional_material_str,
                tool_chain=" -> ".join(tool_chain) if tool_chain else "No tools needed",
                time_context=state.get("time_context") or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            )
            messages = [execution_prompt]
        
        logger.info(f"Tool chain to execute: {' -> '.join(tool_chain) if tool_chain else 'No tools needed'}")

        response: AIMessage = None  # type: ignore
        try:
            response = await self.execution_llm_with_tools.ainvoke([system_prompt] + messages, config=config) # type: ignore
        except Exception as e:
            error_msg = f"Error in tool execution agent: {str(e)}"
            logger.error(error_msg)
            logger.info("Stopping execution due to error.")
            response = AIMessage(content="The execution has been stopped due to an error. Please try again later.")
        
        worker_result = None
        if not response.tool_calls:
            logger.info("✅ TOOL EXECUTION STEP COMPLETED FOR CHAIN")
            logger.info("="*70)

            if last_tool_message is not None and (last_tool_message.name == "entity_augment" or last_tool_message.name == "game_history_retrieval" or last_tool_message.name == "game_info_retrieval"):
                worker_result = last_tool_message.content
            elif response.text:
                worker_result = response.text
            else:
                worker_result = "Worker stopped due to execution error."

            if sub_query:
                semantic_cache.set(sub_query, worker_result, additional_material_list)

            for message in messages:
                if isinstance(message, AIMessage) and message.tool_calls:
                    tool_calls_history.append(message.tool_calls[0])
                if isinstance(message, ToolMessage):
                    tool_results_history.append(message)

        return {
            "messages": messages + [response] if not state.get("messages") else [response], 
            "additional_material": additional_material_list,
            "tool_calls_history": tool_calls_history,
            "tool_results_history": tool_results_history,
            "tool_chain": tool_chain,
            "last_tool_artifact": last_artifact,
            "worker_result": [worker_result] if worker_result else []
        }

    def _check_cache_node(self, state: WorkerState) -> dict:
        """Check semantic cache before executing worker."""
        sub_query = state.get("sub_query")
        if not sub_query:
            return {}
            
        additional_material = state.get("additional_material", [])
        cached_result = semantic_cache.check(sub_query, additional_material)
        
        if cached_result:
            logger.info("⚡ Skipping worker execution due to cache hit (>0.9 similarity).")
            return {
                "worker_result": [cached_result]
            }
        return {}

    def should_execute_worker(self, state: WorkerState):
        """If we already have worker_result from cache hit, skipping worker execution."""
        if state.get("worker_result"):
            return "end"
        return "execute"
    
    def should_continue_call_tool(self, state: WorkerState):
        """Determine whether to continue calling tools or end execution."""
        messages = state.get("messages", [])
        last_message = messages[-1] if len(messages) > 0 else None
        if last_message and isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"    
        else:
            return "end"
