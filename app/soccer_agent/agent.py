import logging
import uuid
import asyncio
from typing import List
from langfuse import get_client

from langchain_core.output_parsers import PydanticOutputParser
from app.soccer_agent.factory.llm_provider import get_llm
from langchain_core.messages import ToolMessage, ToolCall, AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from langgraph.constants import Send

from app.soccer_agent.memory.chat_history import get_postgres_memory
from app.soccer_agent.memory.conversation_memory import CustomSystemPromptMemory
from app.soccer_agent.prompts.agent import get_execution_human_prompt, get_execution_system_prompt, get_planning_prompt_template,get_aggregator_prompt_template
from app.config.config import (SESSION_MEMORY_TOKEN_THRESHOLD, SESSION_MEMORY_RECENT_KEEP)
from app.soccer_agent.memory.session_memory import SessionMemoryManager, SessionMemory
from app.soccer_agent.memory.query_understanding import QueryUnderstandingPipeline
from app.soccer_agent.case_bank.retriever import CaseBankRetriever
from app.soccer_agent.case_bank.cache import case_bank_cache

from app.soccer_agent.toolbox import (
    entity_augment,
    game_history_retrieval,
    game_info_retrieval,
    choice_selection,
    entity_recognition,
    segment,
    frame_selection,
    commentary_generation,
)
from app.schema.soccer_agent.state import PlanningOutput, AgentState, WorkerState
from app.schema.chat import ChatRequest
from app.cache.semantic_cache import semantic_cache

# Configure logging
logger = logging.getLogger(__name__)

class SoccerAgent:
    """
    A LangGraph-based agent for planning soccer knowledge queries.
    
    This agent analyzes questions about soccer and determines the appropriate
    tool chain needed to answer them.
    """
    def __init__(self, checkpointer=None):
        """
        Initialize the Soccer Planning Agent.

        Args:
            checkpointer: Optional LangGraph checkpointer for state persistence
        """
        self.planning_llm = get_llm("planning")
        self.execution_llm = get_llm("execution")
        self.aggregator_llm = get_llm("aggregator")
        self.planning_parser = PydanticOutputParser(pydantic_object=PlanningOutput)
        self.checkpointer = checkpointer
        self.case_bank_retriever = CaseBankRetriever()

        # Session memory + query understanding (use execution_llm: Flash Lite, fast)
        self.session_memory_manager = SessionMemoryManager(
            llm=self.execution_llm,
            token_threshold=SESSION_MEMORY_TOKEN_THRESHOLD,
            recent_messages_to_keep=SESSION_MEMORY_RECENT_KEEP,
        )
        self.query_understanding = QueryUnderstandingPipeline(llm=self.execution_llm)

        # Tool mapping dictionary using LangChain @tool decorated functions
        self.tool_registry: dict[str, BaseTool] = {
            "entity_augment": entity_augment(),
            "game_history_retrieval": game_history_retrieval(),
            "game_info_retrieval": game_info_retrieval(),
            #"entity_recognition": entity_recognition(),
            "choice_selection": choice_selection(),
            "segment": segment(),
            "frame_selection": frame_selection(),
            "commentary_generation": commentary_generation(),
        }

        # List of all tools
        self.tools: List[BaseTool] = list(self.tool_registry.values())
        self.execution_llm_with_tools = self.execution_llm.bind_tools(self.tools) 
        
        self.worker_graph = self._build_worker_graph()
        self.graph = self._build_graph()
        logger.info(f"Loaded {len(self.tools)} LangChain tools")
        
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
        
    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("get_conversation_history", self._get_conversational_history)
        workflow.add_node("understand_user_message", self._understand_user_message)
        workflow.add_node("retrieve_cases_node", self._retrieve_cases_node)
        workflow.add_node("tool_chain_planning", self._tool_chain_planning)
        workflow.add_node("worker_graph", self._worker_node)
        workflow.add_node("aggregator_node", self._aggregator_node)
        workflow.add_node("save_to_memory", self._save_to_memory_node)

        # Define the flow
        workflow.set_entry_point("get_conversation_history")        
        workflow.add_edge("get_conversation_history", "understand_user_message")
        workflow.add_edge("understand_user_message", "retrieve_cases_node")
        workflow.add_edge("retrieve_cases_node", "tool_chain_planning")
        workflow.add_conditional_edges(
            "tool_chain_planning",
            self._trigger_workers,
            ["worker_graph", "aggregator_node"]
        )
        workflow.add_edge("worker_graph", "aggregator_node")
        workflow.add_edge("aggregator_node", "save_to_memory")
        workflow.add_edge("save_to_memory", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def _retrieve_cases_node(self, state: AgentState) -> dict:
        """Retrieve few-shot planning examples from case bank (with Redis cache)."""
        messages = state.get("messages", [])
        user_query = messages[-1].content if messages else ""
        query = state.get("claried_query") or user_query
        has_media = bool(state.get("additional_material"))

        cached = case_bank_cache.get(query, has_media)
        if cached:
            logger.info("CaseBankCache: hit")
            return {"retrieved_cases": cached}

        examples = await self.case_bank_retriever.retrieve(query, has_media)
        if examples:
            case_bank_cache.set(query, has_media, examples)
        return {"retrieved_cases": examples or None}

    async def _worker_node(self, state: WorkerState, config: RunnableConfig) -> dict:
        """
        Wrapper for worker_graph with proper timeout and Langfuse tracing.
        
        Key: Callback cleanup happens in the task's context (where it started),
        timeout wrapper only manages task lifecycle - avoiding OpenTelemetry context errors.
        """
        #
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
        
    async def _get_conversational_history(self, state: AgentState, config: RunnableConfig):
        """
        This node retrieves the conversation history from the storage and updates the state.
        """
        messages = state.get("messages", [])
        user_query = messages[-1].text if messages else ""
        additional_material = state.get("additional_material", [])
        metadata = config.get("metadata", {})
        thread_id = metadata.get("thread_id", str(uuid.uuid4()))

        # --- Flags for tracing metadata ---
        loading_history_status = "failed" # "success"/"failed"
        load_from_cache = False # True: This question has already stored in cache, just load from cache database
        pass_full_history = False # True: The history length is not over threshold limit, give the agent full history without summary
        is_ambigous = False # True: Even thought using history context, the use is ambigous

        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="chain", 
            name="create_history_string") as observation:

            # Load conversation history
            chat_history = None
            if thread_id:
                try:
                    memory_object, connection, pool = await self.get_memory(session_id=thread_id)
                    try:
                        history = memory_object.load_memory_variables({})
                        chat_history = history.get("history")
                        loading_history_status = "success"
                    except Exception as e:
                        logging.warning(f"Memory load failed for session {thread_id}: {e}")
                    finally:
                        await self.cleanup(connection=connection, pool=pool, session_id=thread_id)
                except Exception as e:
                    logging.warning(f"Failed to obtain memory for session {thread_id}: {e}")

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

                    cached_memory = await self.session_memory_manager.load_cached_memory(thread_id)
                    if cached_memory is None:
                        if old_msgs:
                            asyncio.create_task(
                                self.session_memory_manager.background_summarise(thread_id, old_msgs)
                            )
                    else:
                        effective_memory = cached_memory
                        load_from_cache = True

                    history_text = self.session_memory_manager.format_compressed_history(
                        effective_memory, recent_msgs_for_qu
                    )
                    pass_full_history = False
                    logger.info(f"[Path B] token_count={token_count}.")
                else:
                    # Path A: full history fits — pass all as recent context
                    recent_msgs_for_qu = chat_history
                    history_text = raw_history_text
                    logger.info(f"[Path A] token_count={token_count}.")
                    pass_full_history = True

            observation.update(
                output={
                    "chat_history": chat_history or "",
                    "history_text": history_text or "No conversation history.",
                    "recent_msgs_for_qu": recent_msgs_for_qu
                }
            )


        # Emit a tool call event to show the planning step in the UI
        await adispatch_custom_event(
            "manually_emit_tool_call",  # An AG-UI event to trigger tool call visualization without an actual tool execution
            data={
                "id": str(uuid.uuid4()),
                "name": "understand_user_message",
                "args": {}
            },
            config=config
        )
        await asyncio.sleep(0.1)  

        # Always run QueryUnderstanding — handles jargon, abbreviations, pronouns
        qu_output = await self.query_understanding.run(
            user_query=user_query,
            session_memory=effective_memory,
            recent_messages=recent_msgs_for_qu,
        )
        logger.info(f"[QU] clarified='{qu_output.clarified_query}' is_ambiguous={qu_output.is_ambiguous}")

        # Short-circuit: if query is ambiguous, ask for clarification immediately
        # Skip if user attached images/videos — visual context resolves the ambiguity
        if qu_output.is_ambiguous and qu_output.clarifying_questions and not additional_material:
            questions_text = "\n".join(f"- {q}" for q in qu_output.clarifying_questions)
            clarification_response = AIMessage(content=(
                f"Câu hỏi của bạn chưa đủ rõ ràng để tôi trả lời chính xác. "
                f"Bạn có thể làm rõ thêm không?\n{questions_text}"
            ))
            logger.info("[QU] is_ambiguous=True — returning clarification request, skipping graph.")
            
            return Command(
                goto="END",
                update={
                    "messages": messages + [clarification_response],
                }
            )

        # --- Update tracing span metadata ---
        langfuse = get_client()
        langfuse.update_current_span(
            metadata={
                "loading_history_status": loading_history_status,
                "load_from_cache": load_from_cache,
                "pass_full_history": pass_full_history,
                "is_ambigous": qu_output.is_ambiguous
            }
        )
        return {
            "messages": state.get("messages", []), # Copilotkit will append new user messages to "messages", so we need to update the state
            "conversation_history": history_text,
            "claried_query": qu_output.clarified_query,
        }

    async def _understand_user_message(self, state: AgentState, config: RunnableConfig):
        """
        This node is responsible for understanding the user message based on the conversation history. If the message is ambiguous, it will ask for clarification.
        """
        messages = state.get("messages", [])
        user_query = messages[-1].text if messages else ""

        # TODO: Refactor _get_conversational_history seperate between retrieving history and running query understanding
        
        return {}

    async def _tool_chain_planning(self, state: AgentState, config: RunnableConfig) -> dict:
        """
        This node is responsible for planning the tool chain to answer the user's query.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with parsed tool_chain
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
            "manually_emit_tool_call",  # An AG-UI event to trigger tool call visualization without an actual tool execution
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
        
    def _trigger_workers(self, state: AgentState, config: RunnableConfig):
        """Map worker executions for each parallel tool chain."""
        planning_output = state.get("planning_output")
        tool_chains = planning_output.tool_chains if planning_output else []
        sub_queries = planning_output.sub_queries if planning_output else []
        need_call_tools = planning_output.need_call_tools if planning_output else True
        messages = state.get("messages", [])
        claried_query = state.get("claried_query") or messages[-1].content if messages else ""
        
        # Short-circuit to aggregator if planner says no tools needed, or no chains provided
        if not need_call_tools or not tool_chains:
            logger.info(f"⏭️ Skipping workers (need_call_tools={need_call_tools}, tool_chains={tool_chains}). Going straight to aggregator.")
            return "aggregator_node"
            
        sends = []
        for idx, chain in enumerate(tool_chains):
            sub_query = sub_queries[idx] if idx < len(sub_queries) else claried_query
            worker_state = {
                "messages": [], # Start with empty messages for the worker;
                "sub_query": sub_query,
                "additional_material": state.get("additional_material", []),
                "tool_chain": chain,
                "tool_calls_history": [],
                "tool_results_history": [],
                "last_tool_artifact": None,
                "parallel_results": []
            }
            sends.append(Send("worker_graph", worker_state))
        return sends
    
    async def _aggregator_node(self, state: AgentState, config: RunnableConfig) -> dict:
        """Aggregate results from parallel executions and provide the final response."""
        messages = state.get("messages", [])
        user_query = messages[-1].content if messages else ""
        additional_material_list = state.get("additional_material", [])
        additional_material = ", ".join(additional_material_list) if additional_material_list else "None"
        conversation_history = state.get("conversation_history", "No previous conversation.")
        results = state.get("parallel_results", [])

        logger.info("="*70)
        logger.info("🧠 Starting AGGREGATOR STEP")
            
        
        if results:
            worker_results_str = "\n".join([f"Worker {i+1} finding:\n{r}\n" for i, r in enumerate(results)])
        else:
            worker_results_str = "No tools were executed."
            
        aggregator_prompt_template = get_aggregator_prompt_template()
        aggregator_prompt = aggregator_prompt_template.invoke({
            "user_query": state.get("claried_query") or user_query,
            "additional_material": additional_material,
            "conversation_history": conversation_history,
            "worker_results": worker_results_str
        })

        response = await self.aggregator_llm.ainvoke(
            aggregator_prompt,
            config=config
        )
        logger.info("✅ AGGREGATOR STEP COMPLETED")
        logger.info("="*70)
            
        return {
            "messages": [response], # Final response from aggregator
        }
    
    async def _save_to_memory_node(self, state: AgentState, config: RunnableConfig) -> dict:
        """Save the final response along with tool call history into conversation memory."""
        tool_calls_history = state.get("tool_calls_history", [])
        tool_results_history = state.get("tool_results_history", [])
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None # Final response from the aggregator agent.
        metadata = config.get("metadata", {})
        thread_id = metadata.get("thread_id", str(uuid.uuid4()))

        # Find the last user message in the messages history
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break

        try:
            output_with_tools = self._build_tool_summary_for_memory(
                tool_calls_history=tool_calls_history,
                tool_results_history=tool_results_history,
                final_response=last_message.content if last_message else "No response generated",
            )
            asyncio.create_task(
                self._background_save_memory(thread_id, last_user_message.content if last_user_message else "No user message found", output_with_tools)
            )
            # Incremental update of Redis SessionMemory if tool calls were made
            if tool_calls_history:
                asyncio.create_task(
                    self.session_memory_manager.background_update(thread_id, output_with_tools)
                )
        except Exception as e:
            logging.warning(f"Failed to prepare memory save for session {thread_id}: {e}")
        
        return {}

    async def _execution_node(self, state: WorkerState, config: RunnableConfig) -> dict:
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
        messages = state.get("messages", [])

        logger.info(f"🔧 Running TOOL EXECUTION STEP: Step {len(tool_calls_history)}")
        
         # If the previous step is from the tool_node, add the tool execution result to history and add the artifact to state
        last_artifact = state.get("last_tool_artifact")
        last_tool_message = None
        if messages and isinstance(messages[-1], ToolMessage):
            last_tool_message = messages[-1]
            last_artifact = last_tool_message.artifact if hasattr(last_tool_message, 'artifact') else None

        # Create prompt for execution agent
        additional_material_str = ", ".join(additional_material_list) if additional_material_list else "None"
        system_prompt = get_execution_system_prompt()
        if not messages:
            execution_prompt_template = get_execution_human_prompt()
            execution_prompt = execution_prompt_template.format(
                sub_query=sub_query,
                additional_material=additional_material_str,
                tool_chain=" -> ".join(tool_chain) if tool_chain else "No tools needed"
            )
            messages = [execution_prompt]
        
        logger.info(f"Tool chain to execute: {' -> '.join(tool_chain) if tool_chain else 'No tools needed'}")


        # Invoke the execution LLM
        response: AIMessage = None  # type: ignore
        try:
            # Invoke the model with the tool
            response = await self.execution_llm_with_tools.ainvoke([system_prompt] + messages, config=config) # type: ignore
        except Exception as e:
            error_msg = f"Error in tool execution agent: {str(e)}"
            logger.error(error_msg)
            logger.info("Stopping execution due to error.")
            response = AIMessage(content="The execution has been stopped due to an error. Please try again later.")
        
        # If no tool call was made, the execution is completed for this chain. Save the final response and cache it.
        worker_result = None
        if not response.tool_calls:
            logger.info("✅ TOOL EXECUTION STEP COMPLETED FOR CHAIN")
            logger.info("="*70)

            # If the last tool message is from augmentation tool, return the result from the tool instead of the execution agent's response,.
            # Note: this step must be after the execution response to ensure there is no error from the tool.
            if last_tool_message is not None and (last_tool_message.name == "entity_augment" or last_tool_message.name == "game_history_retrieval" or last_tool_message.name == "game_info_retrieval"):
                worker_result = last_tool_message.content
            elif response.text:
                worker_result = response.text
            else:
                worker_result = "Worker stopped due to execution error."

            # Cache the result for this sub-query + additional material combination
            if sub_query:
                semantic_cache.set(sub_query, worker_result, additional_material_list)

            # Build the tool call history and tool result history for compatibility with main graph workflow. 
            # TODO: Remove this if the main graph doesn't need the tool call/result history after the execution step
            for message in messages:
                if isinstance(message, AIMessage) and message.tool_calls:
                    tool_calls_history.append(message.tool_calls[0])  # Assuming one tool call per message for simplicity; adjust if multiple calls are possible
                if isinstance(message, ToolMessage):
                    tool_results_history.append(message)

        return {
            "messages": messages + [response] if not state.get("messages") else [response], # Add the execution agent's response to messages history for the next step's context; if messages is None, initialize with HumanMessage + execution response
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
        """
        Determine whether to continue calling tools or end execution.

        Args:
            state: Current worker state. If there are no tool call, we end execution.

        Returns:
            str: "continue" to keep calling tools, "end" to stop execution.
        """
        messages = state.get("messages", [])
        last_message = messages[-1] if len(messages) > 0 else None
        if last_message and isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"    
        else:
            return "end"
    
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
                result_msg = tool_results_history[i]
                if isinstance(result_msg, dict):
                    content = result_msg.get("content", "")
                    artifact = result_msg.get("artifact")
                else:
                    content = result_msg.content
                    artifact = getattr(result_msg, "artifact", None)
                parts.append(f"    Response: {content}")
                if artifact is not None:
                    parts.append(f"    Artifact: {artifact}")

        parts.append(f"\n[Final Response]\n{final_response}")
        return "\n".join(parts)

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
        Mannually run the agent for a given user query and additional material

        Args:
            request: The ChatRequest object containing user_query and additional_material

        Returns:
            Final response from the agent
        """
        session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, request.user_id))
        config = RunnableConfig(
            metadata={"thread_id": session_id},
        )
        
        with langfuse.start_as_current_observation(
            as_type="chain", 
            name="soccer_agent_request"
        ) as root_trace:
            root_trace.update(input={"user_query": request.user_query})
            try:
                logger.info(f"Starting run for query: {request.user_query[:100]}...")
                initial_state = {
                    "user_query": request.user_query,
                    "claried_query": "",
                    "additional_material": request.additional_material or [],
                    "tool_chains": [],
                    "sub_queries": [],
                    "tool_calls_history": [],
                    "tool_results_history": [],
                    "last_tool_artifact": None,
                    "conversation_history": "",
                    "parallel_results": [],
                    "need_call_tools": True
                }
                logger.info("State initialized")
                
                # Execute the graph with the initial state
                with root_trace.start_as_current_observation(as_type="span", name="execute_langgraph") as graph_span:
                    graph_span.update(input={"initial_state": {k: (v if k != "additional_material" else "List[str]") for k, v in initial_state.items()}})
                    final_state = await self.graph.ainvoke(initial_state, config=config)
                    last_message = final_state.get("messages", [])[-1] if final_state.get("messages") else None
                    graph_span.update(
                        output={
                            "final_response": last_message.text if last_message else "No response generated",
                        }
                    )
                    logger.info("Graph execution completed")
                
                root_trace.update(output={"final_response": last_message.text if last_message else "No response generated"})
                return last_message.text if last_message else "No response generated"

            except Exception as e:
                logging.error(f"Error in agent execution: {e}", exc_info=True)
                root_trace.update(output={"error": str(e)}, level="ERROR")
                return f"✗ Error: {str(e)}"


def get_agent_service() -> SoccerAgent:
    """Get the singleton SoccerAgent service from main.py."""
    import main
    if main.agent_service is None:
        raise RuntimeError("SoccerAgent not initialized. This should not happen if lifespan is working correctly.")
    return main.agent_service