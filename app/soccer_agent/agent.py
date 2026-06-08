import logging
import uuid
import os
from datetime import datetime
from typing import List

# Register custom types for LangGraph Checkpointer
os.environ["LANGGRAPH_ALLOWED_MSGPACK_MODULES"] = "app.schema.soccer_agent.state"

from langfuse import get_client
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage

# Internal Imports
from app.config.settings import settings
from app.schema.soccer_agent.state import AgentState
from app.schema.chat import ChatRequest
from app.soccer_agent.factory.llm_provider import get_llm
from app.soccer_agent.case_bank.retriever import CaseBankRetriever

# Tools
from app.soccer_agent.toolbox import (
    entity_augment,
    entity_recognition,
    game_history_retrieval,
    game_info_retrieval,
    choice_selection,
    segment,
    frame_selection,
    commentary_generation,
    web_news_search,
)

# Nodes
from app.soccer_agent.nodes.conversation_history import ConversationHistoryNode
from app.soccer_agent.nodes.context_retrieval import ContextRetrievalNode
from app.soccer_agent.nodes.unified_planning import UnifiedPlanningNode
from app.soccer_agent.nodes.worker import WorkerNodes
from app.soccer_agent.nodes.aggregator import AggregatorNode
from app.soccer_agent.nodes.memory_saving import SaveToMemoryNode

logger = logging.getLogger(__name__)

class SoccerAgent:
    """
    A LangGraph-based agent for planning and executing soccer knowledge queries.
    Uses Case Bank for few-shot planning and Long-term Memory for knowledge retrieval.
    """
    def __init__(self, checkpointer=None):
        # 1. LLM Initialization
        self.planning_llm = get_llm("planning")
        self.execution_llm = get_llm("execution")
        self.aggregator_llm = get_llm("aggregator")
        self.checkpointer = checkpointer
        
        # 2. Service Initialization
        self.case_bank_retriever = CaseBankRetriever()

        # 3. Tool Registration
        self.tool_registry: dict[str, BaseTool] = {
            "entity_augment": entity_augment(),
            "game_history_retrieval": game_history_retrieval(),
            "game_info_retrieval": game_info_retrieval(),
            "choice_selection": choice_selection(),
            "segment": segment(),
            "frame_selection": frame_selection(),
            "commentary_generation": commentary_generation(),
            "web_news_search": web_news_search(),
            "entity_recognition": entity_recognition(),
        }
        self.tools = list(self.tool_registry.values())
        self.execution_llm_with_tools = self.execution_llm.bind_tools(self.tools) 
        
        # 4. Node Initialization
        self.history_node = ConversationHistoryNode()
        self.context_retrieval_node = ContextRetrievalNode(self.case_bank_retriever)
        self.planning_node = UnifiedPlanningNode(self.planning_llm, self.tools)
        self.worker_nodes = WorkerNodes(self.execution_llm_with_tools, self.tools)
        self.aggregator_node = AggregatorNode(self.aggregator_llm)
        self.memory_saving_node = SaveToMemoryNode()

        # 5. Graph Compilation
        self.graph = self._build_graph()
        logger.info(f"✅ SoccerAgent initialized with {len(self.tools)} tools")
        
    def _build_graph(self) -> CompiledStateGraph:
        """Construct the unified LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add Core Nodes
        workflow.add_node("get_history", self.history_node.get_conversational_history)
        workflow.add_node("context_retrieval", self.context_retrieval_node.retrieve_context_node)
        workflow.add_node("unified_planning", self.planning_node.unified_planning_node)
        workflow.add_node("worker_graph", self.worker_nodes.worker_node)
        workflow.add_node("aggregator", self.aggregator_node.aggregator_node)
        workflow.add_node("save_memory", self.memory_saving_node.save_to_memory_node)

        # Build Parallel Entry
        workflow.add_edge(START, "get_history")
        workflow.add_edge(START, "context_retrieval")

        # Sync into Planning
        workflow.add_edge("get_history", "unified_planning")
        workflow.add_edge("context_retrieval", "unified_planning")
        
        # Execution Path
        workflow.add_conditional_edges(
            "unified_planning",
            self.worker_nodes.trigger_workers,
            ["worker_graph", "aggregator"]
        )
        workflow.add_edge("worker_graph", "aggregator")
        workflow.add_edge("aggregator", "save_memory")
        workflow.add_edge("save_memory", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def run(self, request: ChatRequest) -> str:
        """Execute the agent for a given request with tracing and observability."""
        # thread_id isolates per-session LangGraph state; user_id is for cross-session long-term memory
        thread_id = request.session_id or str(uuid.uuid4())
        config = RunnableConfig(
            configurable={"thread_id": thread_id},
            metadata={"thread_id": thread_id, "user_id": request.user_id},
        )
        
        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="chain", 
            name="soccer_agent_request"
        ) as root_trace:
            root_trace.update(input={"user_query": request.user_query})
            
            try:
                logger.info(f"🚀 Processing Query: {request.user_query[:100]}...")
                initial_state = {
                    "messages": [HumanMessage(content=request.user_query)],
                    "user_query": request.user_query,
                    "clarified_query": "",
                    "additional_material": request.additional_material or [],
                    "planning_output": None,
                    "tool_chains": [],
                    "sub_queries": [],
                    "need_call_tools": True,
                    "pending_clarifications": [],
                    "tool_calls_history": [],
                    "tool_results_history": [],
                    "last_tool_artifact": None,
                    "conversation_history": "",
                    "long_term_context": "",
                    "parallel_results": [],
                    "time_context": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Execute LangGraph
                final_state = await self.graph.ainvoke(initial_state, config=config)
                
                # Extract Final Response
                messages = final_state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    # Handle different message types (AIMessage or text)
                    final_response = getattr(last_msg, 'content', str(last_msg))
                else:
                    final_response = "I'm sorry, I couldn't generate a response."

                root_trace.update(output={"final_response": final_response})
                return final_response

            except Exception as e:
                logger.error(f"❌ Error in agent execution: {e}", exc_info=True)
                root_trace.update(output={"error": str(e)}, level="ERROR")
                return f"✗ Error encountered during processing: {str(e)}"


def get_agent_service() -> SoccerAgent:
    """Singleton getter for the SoccerAgent service."""
    import main
    if main.agent_service is None:
        raise RuntimeError("SoccerAgent not initialized.")
    return main.agent_service