import logging
import uuid
from typing import List
from langfuse import get_client

from langchain_core.output_parsers import PydanticOutputParser
from app.soccer_agent.factory.llm_provider import get_llm
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph, RunnableConfig

from app.config.config import SESSION_MEMORY_TOKEN_THRESHOLD, SESSION_MEMORY_RECENT_KEEP
from app.soccer_agent.memory.session_memory import SessionMemoryManager
from app.soccer_agent.memory.query_understanding import QueryUnderstandingPipeline
from app.soccer_agent.case_bank.retriever import CaseBankRetriever

from app.soccer_agent.toolbox import (
    entity_augment,
    game_history_retrieval,
    game_info_retrieval,
    choice_selection,
    entity_recognition,
    segment,
    frame_selection,
    commentary_generation,
    web_news_search,
)
from app.schema.soccer_agent.state import PlanningOutput, AgentState
from app.schema.chat import ChatRequest

# Import node classes
from app.soccer_agent.nodes.conversation_history import ConversationHistoryNode
from app.soccer_agent.nodes.user_message_understanding import UnderstandUserMessageNode
from app.soccer_agent.nodes.case_retrieval import RetrieveCasesNode
from app.soccer_agent.nodes.tool_planning import ToolChainPlanningNode
from app.soccer_agent.nodes.worker import WorkerNodes
from app.soccer_agent.nodes.aggregator import AggregatorNode
from app.soccer_agent.nodes.memory_saving import SaveToMemoryNode

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
            "web_news_search": web_news_search(),
        }

        # List of all tools
        self.tools: List[BaseTool] = list(self.tool_registry.values())
        self.execution_llm_with_tools = self.execution_llm.bind_tools(self.tools) 
        
        # Initialize node classes
        self.history_node = ConversationHistoryNode(self.session_memory_manager)
        self.understanding_node = UnderstandUserMessageNode(self.query_understanding)
        self.case_retrieval_node = RetrieveCasesNode(self.case_bank_retriever)
        self.planning_node = ToolChainPlanningNode(self.planning_llm, self.planning_parser, self.tools)
        self.worker_nodes = WorkerNodes(self.execution_llm_with_tools, self.tools)
        self.aggregator_node = AggregatorNode(self.aggregator_llm)
        self.memory_saving_node = SaveToMemoryNode(self.session_memory_manager)

        self.graph = self._build_graph()
        logger.info(f"Loaded {len(self.tools)} LangChain tools")
        
    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("get_conversation_history", self.history_node.get_conversational_history)
        workflow.add_node("understand_user_message", self.understanding_node.understand_user_message)
        workflow.add_node("retrieve_cases_node", self.case_retrieval_node.retrieve_cases_node)
        workflow.add_node("tool_chain_planning", self.planning_node.tool_chain_planning)
        workflow.add_node("worker_graph", self.worker_nodes.worker_node)
        workflow.add_node("aggregator_node", self.aggregator_node.aggregator_node)
        workflow.add_node("save_to_memory", self.memory_saving_node.save_to_memory_node)

        # Define the flow
        workflow.set_entry_point("get_conversation_history")        
        workflow.add_edge("get_conversation_history", "understand_user_message")
        workflow.add_edge("understand_user_message", "retrieve_cases_node")
        workflow.add_edge("retrieve_cases_node", "tool_chain_planning")
        workflow.add_conditional_edges(
            "tool_chain_planning",
            self.worker_nodes.trigger_workers,
            ["worker_graph", "aggregator_node"]
        )
        workflow.add_edge("worker_graph", "aggregator_node")
        workflow.add_edge("aggregator_node", "save_to_memory")
        workflow.add_edge("save_to_memory", END)

        return workflow.compile(checkpointer=self.checkpointer)

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
            configurable={"thread_id": session_id},
            metadata={"thread_id": session_id},
        )
        langfuse = get_client()
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