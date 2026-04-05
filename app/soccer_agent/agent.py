import logging
import uuid
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Callable, Annotated, Any
import operator
from pydantic import BaseModel, Field
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import ToolMessage, ToolCall, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.constants import Send

from app.soccer_agent.memory.chat_history import get_postgres_memory
from app.soccer_agent.memory.conversation_memory import CustomSystemPromptMemory
from app.soccer_agent.prompts.agent import get_planning_prompt_template, get_execution_prompt_template,get_aggregator_prompt_template
from app.config.config import (
    DEFAULT_MODEL, GEMINI_2_5_FLASH, GEMINI_2_5_FLASH_LITE, MODEL_TEMPERATURE, MODEL_TOP_P, MAX_COMPLETION_TOKENS,
    SESSION_MEMORY_TOKEN_THRESHOLD, SESSION_MEMORY_RECENT_KEEP,
)
from app.soccer_agent.memory.session_memory import SessionMemoryManager, SessionMemory
from app.soccer_agent.memory.query_understanding import QueryUnderstandingPipeline
from app.config.settings import settings

from app.soccer_agent.toolbox import (
    textual_entity_search, 
    textual_retrieval_augment, 
    game_history_retrieval, 
    game_info_retrieval, 
    game_search, 
    choice_selection,
    entity_recognition,
    segment,
    frame_selection,
    commentary_generation,
)
from app.schema.chat import ChatRequest
from app.cache.semantic_cache import semantic_cache

# Configure logging
logger = logging.getLogger(__name__)
langfuse_client = get_client()

# Structured output models for LLM responses
class PlanningOutput(BaseModel):
    """Structured output for tool chain planning."""
    tool_chains: Optional[List[List[str]]] = Field(default=None, description="A list of lists of EXACT tool names needed to answer the query. Each inner list represents an independent chain of tools that can run in parallel. Tools within an inner list run sequentially.")
    sub_queries: Optional[List[str]] = Field(default=None, description="A list of strings, corresponding to each tool chain in `tool_chains`. Each string should be the specific decomposed part of the user query that the respective tool chain is responsible for answering.")
    need_call_tools: Optional[bool] = Field(default=True, description="Indicates whether tool calls are necessary")

# Define the state structure for the agent
class AgentState(TypedDict):
    """Parent state structure for the planning agent."""
    user_query: str # The user's soccer-related question
    claried_query: str # User query with pronouns/abbreviations resolved by QueryUnderstandingPipeline
    additional_material: Optional[List[str]] # Additional material (e.g, image/video related to the user question)
    tool_chains: List[List[str]] # The planned sequence of tools to execute.
    sub_queries: List[str] # The decomposed sub-queries for each worker.
    parallel_results: Annotated[List[str], operator.add] # Aggregated parallel results
    tool_calls_history: Annotated[List[ToolCall], operator.add] # History of tool calls
    tool_results_history: Annotated[List[ToolMessage], operator.add] # History of tool results
    tool_node_messages: Annotated[List, operator.add] # Messages exchanged
    last_tool_artifact: Optional[Any] # To store the tool's artifact output from the last tool call
    need_call_tools: Optional[bool] # Flag to indicate if more tools need to be called
    conversation_history: Optional[str] # Optional conversation history for context
    final_response: str # Final aggregated response

class WorkerState(TypedDict):
    """State for individual tool chain execution workers."""
    sub_query: str
    additional_material: Optional[List[str]]
    tool_chain: List[str]
    tool_calls_history: List[ToolCall]
    tool_results_history: List[ToolMessage]
    tool_node_messages: List
    parallel_results: List[str]
    last_tool_artifact: Optional[Any]

class SoccerAgent:
    """
    A LangGraph-based agent for planning soccer knowledge queries.
    
    This agent analyzes questions about soccer and determines the appropriate
    tool chain needed to answer them.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the Soccer Planning Agent.
        
        Args:
            model_name: The LLM model to use (default from config)
        """
        self.planning_llm = ChatGoogleGenerativeAI(
            model=GEMINI_2_5_FLASH, 
            api_key=settings.GOOGLE_API_KEY,
            temperature=MODEL_TEMPERATURE, 
            top_p=MODEL_TOP_P,
            max_output_tokens=MAX_COMPLETION_TOKENS,
            thinking_budget=3000,
            include_thoughts=True #type: ignore
        )
        self.execution_llm = ChatGoogleGenerativeAI(
            model=GEMINI_2_5_FLASH_LITE,
            api_key=settings.GOOGLE_API_KEY,
            temperature=MODEL_TEMPERATURE,
            top_p=MODEL_TOP_P,
            max_output_tokens=MAX_COMPLETION_TOKENS,
            thinking_budget=4000,
            include_thoughts=True #type: ignore
        )
        self.planning_parser = PydanticOutputParser(pydantic_object=PlanningOutput)

        # Session memory + query understanding (use execution_llm: Flash Lite, fast)
        self.session_memory_manager = SessionMemoryManager(
            llm=self.execution_llm,
            token_threshold=SESSION_MEMORY_TOKEN_THRESHOLD,
            recent_messages_to_keep=SESSION_MEMORY_RECENT_KEEP,
        )
        self.query_understanding = QueryUnderstandingPipeline(llm=self.execution_llm)

        # Tool mapping dictionary using LangChain @tool decorated functions
        self.tool_registry: dict[str, BaseTool] = {
            "textual_entity_search": textual_entity_search(),
            "textual_retrieval_augment": textual_retrieval_augment(),
            "game_search": game_search(),
            "game_history_retrieval": game_history_retrieval(),
            "game_info_retrieval": game_info_retrieval(),
            "entity_recognition": entity_recognition(),
            "choice_selection": choice_selection(),
            "segment": segment(),
            # "frame_selection": frame_selection(),
            "commentary_generation": commentary_generation(),
        }

        # List of all tools
        self.tools: List[BaseTool] = list(self.tool_registry.values())
        self.execution_llm_with_tools = self.execution_llm.bind_tools(self.tools) 
        
        self.worker_graph = self._build_worker_graph()
        self.graph = self._build_graph()
        logger.info(f"SoccerAgent initialized with model: {model_name}")
        logger.info(f"Loaded {len(self.tools)} LangChain tools")
        
    def _build_worker_graph(self) -> CompiledStateGraph:
        """Build the LangGraph worker workflow."""
        workflow = StateGraph(WorkerState)
        tool_node = ToolNode(self.tools, messages_key="tool_node_messages")
        
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
        
    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("tool_chain_planning", self._tool_chain_planning)
        workflow.add_node("worker_graph", self._worker_node)
        workflow.add_node("aggregator_node", self._aggregator_node)

        # Define the flow
        workflow.set_entry_point("tool_chain_planning")
        
        workflow.add_conditional_edges(
            "tool_chain_planning", 
            self._trigger_workers, 
            ["worker_graph", "aggregator_node"]
        )
        workflow.add_edge("worker_graph", "aggregator_node")
        workflow.add_edge("aggregator_node", END)
        
        return workflow.compile()
    
    @observe(name="worker_node", as_type="span", capture_input=True)
    async def _worker_node(self, state: WorkerState) -> dict:
        """
        Wrapper for worker_graph with proper timeout and Langfuse tracing.
        
        Key: Callback cleanup happens in the task's context (where it started),
        timeout wrapper only manages task lifecycle - avoiding OpenTelemetry context errors.
        """
        try:
            async def worker_execution():
                # Initialize handler inside the task's context
                langfuse_handler = CallbackHandler()
                
                return await asyncio.wait_for(
                    self.worker_graph.ainvoke(
                        state,
                        config={
                            "callbacks": [langfuse_handler],
                            "metadata": {
                                "sub_query": state.get("sub_query", "")[:100],
                                "tool_chain": ", ".join(state.get("tool_chain", [])),
                                "worker_id": id(state)
                            }
                        }
                    ),
                    timeout=120.0
                )
        
            result = await worker_execution()
            
            return {
                "parallel_results": result.get("parallel_results", []),
                "tool_calls_history": result.get("tool_calls_history", []),
                "tool_results_history": result.get("tool_results_history", []),
                "tool_node_messages": result.get("tool_node_messages", [])
            }
        except asyncio.TimeoutError:
            error_msg = f"[Timeout] Worker for sub-query '{state.get('sub_query', 'unknown')}' exceeded 120 seconds."
            logger.error(error_msg)
            return {
                "parallel_results": [error_msg],
                "tool_calls_history": [],
                "tool_results_history": [],
                "tool_node_messages": []
            }
        except Exception as e:
            error_msg = f"[Error] Worker for sub-query '{state.get('sub_query', 'unknown')}' failed with error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "parallel_results": [error_msg],
                "tool_calls_history": [],
                "tool_results_history": [],
                "tool_node_messages": []
            }
    
    async def _tool_chain_planning(self, state: AgentState) -> AgentState:
        """
        This node is responsible for planning the tool chain to answer the user's query.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with parsed tool_chain
        """
        
        with langfuse_client.start_as_current_observation(as_type="span", name="tool_chain_planning") as root_span:
            logger.info("="*70)
            logger.info("🧠 Starting TOOL CHAIN PLANNING STEP")
            additional_material_list = state.get("additional_material")
            additional_material = ", ".join(additional_material_list) if additional_material_list else "None"
            conversation_history = state.get("conversation_history", "No previous conversation.")
            
            # Create prompt for planning agent
            with root_span.start_as_current_observation(as_type="span", name="create_prompt") as prompt_span:
                tool_descriptions = ""
                for tool in self.tools:
                    tool_descriptions += f"- {tool.name}: {tool.description}\n"
                
                format_instructions = self.planning_parser.get_format_instructions()
                planning_agent_prompt_template = get_planning_prompt_template()
                planning_agent_prompt = planning_agent_prompt_template.invoke({
                    "toolbox_descriptions": tool_descriptions,
                    "format_instructions": format_instructions,
                    "user_query": state.get("claried_query") or state["user_query"],
                    "additional_material": additional_material,
                    "conversation_history": conversation_history
                })
                
                prompt_span.update(
                    output={"planning_prompt": planning_agent_prompt},
                    metadata={"user_query": state["user_query"][:100]}
                )
            
            # Call the planning model to get the tool chain
            langfuse_handler = CallbackHandler()
            response = await self.planning_llm.ainvoke(
                planning_agent_prompt,
                config={
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "user_query": state["user_query"][:100],
                        "model_step": "planning"
                    }
                }
            )
            response_text = response.text if hasattr(response, 'text') else str(response)
                
            planning_output: PlanningOutput = self.planning_parser.parse(response_text)
            logger.debug(f"🤖 Response from Planning Agent: {planning_output}")
            
            # Update root span with final output
            root_span.update(
                output={
                    "tool_chains": planning_output.tool_chains,
                    "sub_queries": planning_output.sub_queries
                }
            )
            
            logger.info("Tool Chain Planning Results:")
            logger.info(f"\t Tool Chains: {planning_output.tool_chains}")
            logger.info("✅ TOOL CHAIN PLANNING STEP COMPLETED")
            logger.info("="*70)

        return {
            "user_query": state["user_query"],
            "additional_material": sorted(state.get("additional_material") or []),
            "claried_query": state.get("claried_query") or state["user_query"],
            "tool_chains": planning_output.tool_chains or [],
            "sub_queries": planning_output.sub_queries or [],
            "parallel_results": state.get("parallel_results", []),
            "tool_calls_history": state.get("tool_calls_history", []),
            "tool_results_history": state.get("tool_results_history", []),
            "tool_node_messages": state.get("tool_node_messages", []),
            "last_tool_artifact": state.get("last_tool_artifact"),
            "need_call_tools": planning_output.need_call_tools,
            "conversation_history": state.get("conversation_history", ""),
            "final_response": state.get("final_response", "")
        }
        
    def _trigger_workers(self, state: AgentState):
        """Map worker executions for each parallel tool chain."""
        tool_chains = state.get("tool_chains", [])
        sub_queries = state.get("sub_queries") or []
        need_call_tools = state.get("need_call_tools", True)
        
        # Short-circuit to aggregator if planner says no tools needed, or no chains provided
        if not need_call_tools or not tool_chains:
            logger.info(f"⏭️ Skipping workers (need_call_tools={need_call_tools}, tool_chains={tool_chains}). Going straight to aggregator.")
            return "aggregator_node"
            
        claried_query = state.get("claried_query") or state["user_query"]
        sends = []
        for idx, chain in enumerate(tool_chains):
            sub_query = sub_queries[idx] if idx < len(sub_queries) else claried_query
            worker_state = {
                "sub_query": sub_query,
                "additional_material": state.get("additional_material", []),
                "tool_chain": chain,
                "tool_calls_history": [],
                "tool_results_history": [],
                "tool_node_messages": [],
                "last_tool_artifact": None,
                "parallel_results": []
            }
            sends.append(Send("worker_graph", worker_state))
        return sends
    
    async def _execution_node(self, state: WorkerState) -> dict:
        """
        Iteratively execute the tool chain step by step using bind_tools with tool_choice.
        This allows the LLM to see the full tool schema and automatically generate correct parameters.
        
        Args:
            state: Current worker state with tool_chain
            
        Returns:
            Updated worker dict
        """
        sub_query = state.get("sub_query")
        additional_material_list = state.get("additional_material", [])
        tool_chain = state["tool_chain"]
        tool_calls_history = state.get("tool_calls_history", [])
        tool_results_history = state.get("tool_results_history", [])
        tool_node_messages = state.get("tool_node_messages", [])

        logger.info(f"🔧 Running TOOL EXECUTION STEP: Step {len(tool_calls_history)}")
        
         # If the previous step is from the tool_node, add the tool execution result to history and add the artifact to state
        last_artifact = state.get("last_tool_artifact")
        if tool_node_messages:
            tool_result: ToolMessage = tool_node_messages[-1] # Each time only one tool is called, so the last message is the result of the current tool
            tool_results_history.append(tool_result)
            last_artifact = tool_result.artifact if hasattr(tool_result, 'artifact') else None
            logger.info(f"Received tool result: {tool_result.content}")

        # Format List[str] to string for prompt
        additional_material_str = ", ".join(additional_material_list) if additional_material_list else "None"
        
        # Build execution history string and prompt
        history_str = self._build_history_string(tool_calls_history, tool_results_history)
        execution_prompt_template = get_execution_prompt_template()
        execution_prompt = execution_prompt_template.invoke({
            "sub_query": sub_query,
            "additional_material": additional_material_str,
            "tool_chain": " -> ".join(tool_chain) if tool_chain else "No tools needed",
            "history": history_str,
            
        })
        logger.info(f"Tool chain to execute: {' -> '.join(tool_chain) if tool_chain else 'No tools needed'}")


        # 🔴 Fix: initialize to None so it's always bound, even if invoke() throws
        response: AIMessage = None  # type: ignore
        try:
            # Invoke the model with the tool
            response = await self.execution_llm_with_tools.ainvoke(execution_prompt) # type: ignore
            logger.info(f"🤖 Response from execution agent: \n \t Response content: {response.text} \n \t Tool Calls: {response.tool_calls}")
            tool_node_messages = [response] # Add the message to tool_node_messages for tool_node if there is no tool call the should_or_continue node will end execution
            
            if response.tool_calls:
                # Extract the tool call and add to tool calls history
                tool_call: ToolCall = response.tool_calls[0]
                tool_calls_history.append(tool_call)
                logger.info(f"Added ToolCall to history. Executed total: {len(tool_calls_history)}")
            else:
                logger.info("No tool call made by the execution agent. Ending execution.")
    
        except Exception as e:
            error_msg = f"Error in tool execution agent: {str(e)}"
            logger.error(error_msg)
            logger.info("Stopping execution due to error.")
            response = AIMessage(content="The execution has been stopped due to an error. Please try again later.")
            tool_node_messages = [response] 
        
        # Update state
        base_state = {
            "additional_material": additional_material_list,
            "tool_calls_history": tool_calls_history,
            "tool_results_history": tool_results_history,
            "tool_chain": tool_chain,
            "tool_node_messages": tool_node_messages,
            "last_tool_artifact": last_artifact
        }
        
        # 🔴 Fix: use getattr to safely check response.tool_calls (response may be None after an exception)
        if not getattr(response, 'tool_calls', None):
            logger.info("✅ TOOL EXECUTION STEP COMPLETED FOR CHAIN")
            logger.info("="*70)
            if response is not None:
                result_text = getattr(response, 'text', None) or response.content
            else:
                result_text = "Worker stopped due to execution error."
            base_state["parallel_results"] = [result_text]
            if sub_query:
                semantic_cache.set(sub_query, result_text, additional_material_list)

        return base_state

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
                "parallel_results": [cached_result]
            }
        return {}

    def should_execute_worker(self, state: WorkerState):
        """If we already have parallel_results from cache hit, skipping worker execution."""
        if state.get("parallel_results"):
            return "end"
        return "execute"
    
    async def _aggregator_node(self, state: AgentState) -> dict:
        """Aggregate results from parallel executions and provide the final response."""
        with langfuse_client.start_as_current_observation(as_type="span", name="aggregator") as root_span:
            logger.info("="*70)
            logger.info("🧠 Starting AGGREGATOR STEP")
            
            additional_material_list = state.get("additional_material", [])
            additional_material = ", ".join(additional_material_list) if additional_material_list else "None"
            conversation_history = state.get("conversation_history", "No previous conversation.")
            root_span.update(input={
                "additional_material": additional_material,
                "conversation_history": conversation_history
            })

            results = state.get("parallel_results", [])
            if results:
                worker_results_str = "\n".join([f"Worker {i+1} finding:\n{r}\n" for i, r in enumerate(results)])
            else:
                worker_results_str = "No tools were executed."
            
            # Prepare the aggregation context for the prompt
            with root_span.start_as_current_observation(as_type="span", name="prepare_aggregation_context") as prep_span:
                prep_span.update(
                    output={
                        "worker_results_str": worker_results_str,  
                        "num_results": len(results),
                        "has_results": len(results) > 0
                    }
                )
            
                aggregator_prompt_template = get_aggregator_prompt_template()
                aggregator_prompt = aggregator_prompt_template.invoke({
                    "user_query": state.get("claried_query") or state["user_query"],
                    "additional_material": additional_material,
                    "conversation_history": conversation_history,
                    "worker_results": worker_results_str
                })
                prep_span.update(output={"aggregator_prompt": aggregator_prompt})
                
            langfuse_handler = CallbackHandler()
            response = await self.execution_llm.ainvoke(
                aggregator_prompt,
                config={
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "user_query": state["user_query"][:100],
                        "num_worker_results": len(results),
                        "model_step": "aggregation"
                    }
                }
            )
            final_text = response.text if hasattr(response, 'text') else str(response)
            logger.info("✅ AGGREGATOR STEP COMPLETED")
            logger.info("="*70)
            
            return {"final_response": final_text}
    
    def should_continue_call_tool(self, state: WorkerState):
        """
        Determine whether to continue calling tools or end execution.

        Args:
            state: Current worker state. If there are no tool call, we end execution.

        Returns:
            str: "continue" to keep calling tools, "end" to stop execution.
        """
        tool_node_messages = state.get("tool_node_messages")
        last_message = tool_node_messages[-1] if len(tool_node_messages) > 0 else None
        if not last_message or not last_message.tool_calls:
            return "end"
        else:
            return "continue"
    
    def _build_tool_summary_for_memory(
        self,
        tool_calls_history: List[ToolCall],
        tool_results_history: List[ToolMessage],
        final_response: str
    ) -> str:
        """
        Build a structured string combining tool call details and final response
        to be saved into conversation memory for a single chat turn.

        Each tool entry includes: tool_name, input args, response content, and artifact (if any).
        """
        if not tool_calls_history:
            return final_response

        parts = ["[Tool Usage]"]
        for i, tool_call in enumerate(tool_calls_history):
            tool_name = tool_call.get("name", "unknown")
            args = tool_call.get("args", {})
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            parts.append(f"  Step {i + 1}: {tool_name}({args_str})")

            if i < len(tool_results_history):
                result_msg: ToolMessage = tool_results_history[i]
                parts.append(f"    Response: {result_msg.content}")
                artifact = getattr(result_msg, "artifact", None)
                if artifact is not None:
                    parts.append(f"    Artifact: {artifact}")

        parts.append(f"\n[Final Response]\n{final_response}")
        return "\n".join(parts)

    def _build_history_string(self, tool_calls_history: List[ToolCall], tool_results_history: List[ToolMessage]) -> str:
        """
        Build the execution history string for the prompt.
        
        Args:
            tool_calls_history: List of tool calls
            tool_results_history: List of tool results
            
        Returns:
            str: Formatted history string
        """
        if not tool_calls_history and not tool_results_history:
            return "No execution history yet. This is the first step."
        
        history_parts = []

        # Integrate tool calls and results in order
        for i in range(len(tool_calls_history)):
            # Add tool call
            tool_call = tool_calls_history[i]
            tool_name = tool_call.get('name', 'unknown')
            args_items = tool_call.get('args', {})
            args_str = ", ".join(f"{arg}={value}" for arg, value in args_items.items())
            history_parts.append(f"<CALL_TOOL><TOOL_NAME>{tool_name}</TOOL_NAME><TOOL_ARGS>{args_str}</TOOL_ARGS></CALL_TOOL>")
            
            # Add corresponding tool result if available
            if i < len(tool_results_history):
                tool_result = tool_results_history[i]
                history_parts.append(f"<TOOL_RESULT>{tool_result.content}</TOOL_RESULT>")
        
        return "\n".join(history_parts)
    
    async def get_memory(self, session_id: str):
        """
        Get conversation memory for the given session.
        
        Args:
            session_id: Session ID to retrieve memory for
            
        Returns:
            Tuple of (memory_object, connection, pool) for cleanup
        """
        memory, connection, pool = get_postgres_memory(session_id)
        memory_object = CustomSystemPromptMemory(
            memory_key="history",
            chat_memory=memory.chat_memory,
            return_messages=True,
            max_history=15,  # Limit to last 15 messages
        )
        return memory_object, connection, pool
    
    async def cleanup(self, connection, pool, session_id: str):
        """
        Cleanup database connection and return to pool.
        
        Args:
            connection: Database connection to cleanup
            pool: Connection pool to return connection to
            session_id: Session ID for logging
        """
        try:
            if not connection:
                return

            # Try to commit any pending transactions
            try:
                closed = getattr(connection, "closed", False)
                if not closed:
                    connection.commit()
            except Exception:
                # Best-effort commit; ignore errors here
                pass

            # Return connection to pool if available, otherwise close it
            if pool:
                try:
                    pool.putconn(connection)
                    logging.debug(f"Connection returned to pool for session {session_id}")
                except Exception:
                    try:
                        connection.close()
                    except Exception:
                        pass
            else:
                try:
                    connection.close()
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
            try:
                if connection:
                    connection.close()
            except Exception:
                pass

    async def _background_save_memory(
        self, session_id: str, user_query: str, output_with_tools: str
    ) -> None:
        """Save conversation memory in background with a dedicated DB connection."""
        try:
            mem, conn, pool = get_postgres_memory(session_id)
            mem_obj = CustomSystemPromptMemory(
                memory_key="history",
                chat_memory=mem.chat_memory,
                return_messages=True,
                max_history=15,
            )
            await asyncio.to_thread(
                mem_obj.save_context,
                {"input": user_query},
                {"output": output_with_tools},
            )
            if conn and pool:
                conn.commit()
                pool.putconn(conn)
        except Exception as e:
            logger.warning(f"[BackgroundSave] Failed for session {session_id}: {e}")

    async def run(self, request: ChatRequest) -> str:
        """
        Run the complete workflow: planning + execution.

        Args:
            request: The ChatRequest object containing user_query and additional_material

        Returns:
            Final response from the agent
        """
        session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, request.user_id))
        
        with langfuse_client.start_as_current_observation(
            as_type="chain", 
            name="soccer_agent_request"
        ) as root_trace:
            root_trace.update(input={"user_query": request.user_query})
            
            connection = None
            pool = None
            memory_object = None
            
            try:
                logger.info(f"Starting run for query: {request.user_query[:100]}...")
                
                # Load conversation history
                with root_trace.start_as_current_observation(as_type="span", name="load_memory") as mem_span:
                    chat_history = None
                    if session_id:
                        try:
                            memory_object, connection, pool = await self.get_memory(session_id=session_id)
                            try:
                                history = memory_object.load_memory_variables({})
                                chat_history = history.get("history")
                                mem_span.update(output={"memory_loaded": True, "history_length": len(chat_history) if chat_history else 0})
                            except Exception as e:
                                logging.warning(f"Memory load failed for session {session_id}: {e}")
                                mem_span.update(output={"memory_loaded": False, "error": str(e)})
                        except Exception as e:
                            logging.warning(f"Failed to obtain memory for session {session_id}: {e}")
                            mem_span.update(output={"memory_loaded": False, "error": str(e)})

                # Build history and run Query Understanding on all paths
                history_text = ""
                effective_memory = SessionMemory()
                recent_msgs_for_qu = []

                if chat_history:
                    raw_history_text = "\n\n### LỊCH SỬ HỘI THOẠI:\n"
                    for msg in chat_history[-10:]:
                        if hasattr(msg, 'content'):
                            role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                            raw_history_text += f"{role}: {msg.content}\n"

                    token_count = self.session_memory_manager.count_tokens(raw_history_text)

                    if token_count > SESSION_MEMORY_TOKEN_THRESHOLD:
                        # Path B: history exceeds threshold — use SessionMemory + recent slice
                        old_msgs = chat_history[:-(SESSION_MEMORY_RECENT_KEEP-1)]
                        recent_msgs_for_qu = chat_history[-SESSION_MEMORY_RECENT_KEEP:]

                        cached_memory = await self.session_memory_manager.load_cached_memory(session_id)
                        if cached_memory is None:
                            if old_msgs:
                                asyncio.create_task(
                                    self.session_memory_manager.background_summarise(session_id, old_msgs)
                                )
                        else:
                            effective_memory = cached_memory

                        history_text = self.session_memory_manager.format_compressed_history(
                            effective_memory, recent_msgs_for_qu
                        )
                        logger.info(f"[Path B] token_count={token_count}.")
                    else:
                        # Path A: full history fits — pass all as recent context
                        recent_msgs_for_qu = chat_history
                        history_text = raw_history_text
                        logger.info(f"[Path A] token_count={token_count}.")

                # Always run QueryUnderstanding — handles jargon, abbreviations, pronouns
                qu_output = await self.query_understanding.run(
                    user_query=request.user_query,
                    session_memory=effective_memory,
                    recent_messages=recent_msgs_for_qu,
                )
                logger.info(f"[QU] clarified='{qu_output.clarified_query}' is_ambiguous={qu_output.is_ambiguous}")

                # Short-circuit: if query is ambiguous, ask for clarification immediately
                # Skip if user attached images/videos — visual context resolves the ambiguity
                has_media = bool(request.additional_material)
                if qu_output.is_ambiguous and qu_output.clarifying_questions and not has_media:
                    questions_text = "\n".join(f"- {q}" for q in qu_output.clarifying_questions)
                    clarification_response = (
                        f"Câu hỏi của bạn chưa đủ rõ ràng để tôi trả lời chính xác. "
                        f"Bạn có thể làm rõ thêm không?\n{questions_text}"
                    )
                    logger.info("[QU] is_ambiguous=True — returning clarification request, skipping graph.")
                    root_trace.update(output={"final_response": clarification_response, "is_ambiguous": True})
                    return clarification_response

                claried_query = qu_output.clarified_query

                initial_state = {
                    "user_query": request.user_query,
                    "claried_query": claried_query,
                    "additional_material": request.additional_material or [],
                    "tool_chains": [],
                    "sub_queries": [],
                    "tool_calls_history": [],
                    "tool_results_history": [],
                    "tool_node_messages": [],
                    "last_tool_artifact": None,
                    "conversation_history": history_text,
                    "parallel_results": [],
                    "need_call_tools": True
                }
                logger.info("State initialized")
                
                # Execute the graph with the initial state
                with root_trace.start_as_current_observation(as_type="span", name="execute_langgraph") as graph_span:
                    graph_span.update(input={"initial_state": {k: (v if k != "additional_material" else "List[str]") for k, v in initial_state.items()}})
                    final_state = await self.graph.ainvoke(initial_state)
                    graph_span.update(
                        output={
                            "final_response": final_state.get("final_response", "No response generated"),
                        }
                    )
                    logger.info("Graph execution completed")

                result = {
                    "user_query": request.user_query,
                    "tool_chains": final_state.get("tool_chains", []),
                    "tool_calls_history": final_state.get("tool_calls_history", []),
                    "tool_results_history": final_state.get("tool_results_history", []),
                    "final_response": final_state.get("final_response", "No response generated")
                }

                
                # Save conversation memory in background (non-blocking, uses its own DB connection)
                try:
                    output_with_tools = self._build_tool_summary_for_memory(
                        tool_calls_history=result.get("tool_calls_history", []),
                        tool_results_history=result.get("tool_results_history", []),
                        final_response=result["final_response"],
                    )
                    asyncio.create_task(
                        self._background_save_memory(session_id, request.user_query, output_with_tools)
                    )
                    # Incremental update of Redis SessionMemory if tool calls were made
                    if result.get("tool_calls_history"):
                        asyncio.create_task(
                            self.session_memory_manager.background_update(session_id, output_with_tools)
                        )
                except Exception as e:
                    logging.warning(f"Failed to prepare memory save for session {session_id}: {e}")
                logger.info("Run completed successfully")

                root_trace.update(output={"final_response": result["final_response"]})
                return result["final_response"]

            except Exception as e:
                logging.error(f"Error in agent execution: {e}", exc_info=True)
                root_trace.update(output={"error": str(e)}, level="ERROR")
                return f"✗ Error: {str(e)}"
            finally:
                # Cleanup connection
                try:
                    await self.cleanup(connection=connection, pool=pool, session_id=session_id)
                    logging.debug("✅ ChatAgent connection cleaned up")
                except Exception as cleanup_error:
                    logging.error(f"❌ ChatAgent cleanup error: {cleanup_error}")

# Do NOT create singleton here - it will be created in main.py lifespan
# agent_service = SoccerAgent()

def get_agent_service() -> SoccerAgent:
    """Get the singleton SoccerAgent service from main.py."""
    import main
    if main.agent_service is None:
        raise RuntimeError("SoccerAgent not initialized. This should not happen if lifespan is working correctly.")
    return main.agent_service