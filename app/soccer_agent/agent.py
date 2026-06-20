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
from app.soccer_agent.nodes.guardrail import GuardrailNode

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
        self.guardrail_llm = get_llm("guardrail")
        self.checkpointer = checkpointer
        
        # 2. Service Initialization
        self.case_bank_retriever = CaseBankRetriever()

        # 3. Tool Registration
        self.tool_registry: dict[str, BaseTool] = {
            "entity_augment": entity_augment(),
            "game_history_retrieval": game_history_retrieval(),
            "game_info_retrieval": game_info_retrieval(),
            "commentary_generation": commentary_generation(),
            "web_news_search": web_news_search(),
        }
        # entity_recognition probes its Azure endpoint at construction time —
        # an unreachable endpoint must disable this one tool, not the whole agent.
        try:
            self.tool_registry["entity_recognition"] = entity_recognition()
        except Exception as e:
            logger.warning(f"⚠️ entity_recognition unavailable at startup, tool disabled: {e}")
        self.tools = list(self.tool_registry.values())
        self.execution_llm_with_tools = self.execution_llm.bind_tools(self.tools) 
        
        # 4. Node Initialization
        self.history_node = ConversationHistoryNode()
        self.context_retrieval_node = ContextRetrievalNode(self.case_bank_retriever)
        self.planning_node = UnifiedPlanningNode(self.planning_llm, self.tools)
        self.worker_nodes = WorkerNodes(self.execution_llm_with_tools, self.tools)
        self.aggregator_node = AggregatorNode(self.aggregator_llm)
        self.memory_saving_node = SaveToMemoryNode()
        self.guardrail_node = GuardrailNode(self.guardrail_llm)

        # 5. Graph Compilation
        self.graph = self._build_graph()
        logger.info(f"✅ SoccerAgent initialized with {len(self.tools)} tools")

    async def warmup(self) -> None:
        """Warm up all internal services so the first real request has no cold-start overhead.

        Sends each LLM role its system prompt with max_completion_tokens=0:
        establishes HTTP connection pool + primes prompt cache, zero output tokens.
        Non-fatal: failures are logged as warnings, server still starts.
        """
        import asyncio
        from langchain_core.messages import HumanMessage as _HM
        from app.soccer_agent.factory.llm_provider import get_llm
        from app.soccer_agent.prompts.agent import (
            get_unified_planning_prompt_template,
            get_execution_system_prompt,
            get_aggregator_prompt_template,
            get_guardrail_prompt_template,
        )
        from app.soccer_agent.prompts.toolbox.textual_retrieval_augment import (
            get_textual_retrieval_augment_prompt_template,
        )
        from app.soccer_agent.prompts.toolbox.commentary_generation import (
            get_commentary_generation_prompt_template,
        )

        # Core graph roles
        planning_system = get_unified_planning_prompt_template().messages[0]
        execution_system = get_execution_system_prompt()
        aggregator_system = get_aggregator_prompt_template().messages[0]
        guardrail_system = get_guardrail_prompt_template().messages[0]
        # Tool roles (shared router → warming the role warms every tool on it):
        #   retrieval-augment ← entity_augment + game_info/history_retrieval
        #   tool              ← commentary_generation
        retrieval_augment_system = get_textual_retrieval_augment_prompt_template().messages[0]
        tool_system = get_commentary_generation_prompt_template().messages[0]

        async def _warm(name: str, llm, system_msg) -> None:
            try:
                await llm.ainvoke(
                    [system_msg, _HM(content="warmup")],
                    max_completion_tokens=0,
                )
                logger.info(f"✅ {name} LLM warmed up")
            except Exception as e:
                logger.warning(f"⚠️ {name} LLM warmup failed (non-fatal): {e}")

        await asyncio.gather(
            self.case_bank_retriever.warmup(),
            _warm("planning", self.planning_llm, planning_system),
            _warm("execution", self.execution_llm, execution_system),
            _warm("aggregator", self.aggregator_llm, aggregator_system),
            _warm("guardrail", self.guardrail_llm, guardrail_system),
            _warm("retrieval-augment", get_llm("retrieval-augment"), retrieval_augment_system),
            _warm("tool", get_llm("tool"), tool_system),
        )
        logger.info("✅ SoccerAgent warmup complete — all services ready")
        
    def _build_graph(self) -> CompiledStateGraph:
        """Construct the unified LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add Core Nodes
        workflow.add_node("get_history", self.history_node.get_conversational_history)
        workflow.add_node("context_retrieval", self.context_retrieval_node.retrieve_context_node)
        workflow.add_node("guardrail_classify", self.guardrail_node.classify_node)
        workflow.add_node("guardrail_gate", self.guardrail_node.gate_node)
        workflow.add_node("guardrail_refusal", self.guardrail_node.refusal_node)
        workflow.add_node("unified_planning", self.planning_node.unified_planning_node)
        workflow.add_node("worker_graph", self.worker_nodes.worker_node)
        workflow.add_node("aggregator", self.aggregator_node.aggregator_node)
        workflow.add_node("save_memory", self.memory_saving_node.save_to_memory_node)

        # Build Parallel Entry — guardrail runs alongside history + context retrieval
        workflow.add_edge(START, "get_history")
        workflow.add_edge(START, "context_retrieval")
        workflow.add_edge(START, "guardrail_classify")

        # Fan-in: all three parallel branches converge on the guardrail gate
        workflow.add_edge("get_history", "guardrail_gate")
        workflow.add_edge("context_retrieval", "guardrail_gate")
        workflow.add_edge("guardrail_classify", "guardrail_gate")

        # Gate: off-topic hard-stops to the canned refusal, skipping planning/workers/aggregator
        workflow.add_conditional_edges(
            "guardrail_gate",
            self.guardrail_node.gate_router,
            {"proceed": "unified_planning", "blocked": "guardrail_refusal"}
        )
        workflow.add_edge("guardrail_refusal", END)

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
                    "additional_material": request.additional_material or {},
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
                    "time_context": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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