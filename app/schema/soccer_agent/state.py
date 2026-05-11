from typing import Annotated, Any, Dict, List, Optional, TypedDict
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage, ToolCall
from copilotkit import CopilotKitState

# Structured output models for LLM responses
class PlanningOutput(BaseModel):
    """Structured output for tool chain planning."""
    tool_chains: Optional[List[List[str]]] = Field(default=None, description="List of independent tool chains to answer the query")
    sub_queries: Optional[List[str]] = Field(default=None, description="List of specific decomposed sub-queries, each corresponding to a tool chain")
    need_call_tools: Optional[bool] = Field(default=True, description="Indicates whether tool calls are necessary")

# Define the state structure for the agent
class AgentState(CopilotKitState):
    """Parent state structure for the planning agent. It is derived from CopilotKitState, which provides 'messages' list to store the conversation history"""
    claried_query: str # User query with pronouns/abbreviations resolved by QueryUnderstandingPipeline
    additional_material: Optional[List[str]] # Additional material (e.g, image/video related to the user question)
    planning_output: Optional[PlanningOutput] # The output from the planning step, which includes tool chains and sub-queries
    parallel_results: Annotated[List[str], operator.add] # Aggregated parallel results
    tool_calls_history: Annotated[List[ToolCall], operator.add] # History of tool calls
    tool_results_history: Annotated[List[ToolMessage], operator.add] # History of tool results
    conversation_history: Optional[str] # Optional conversation history for context
    retrieved_cases: Optional[List[str]] # Few-shot planning examples retrieved from case bank
    recent_msgs_for_qu: Optional[List[Any]] # Passed from history to understanding
    effective_memory: Optional[Any] # Passed from history to understanding

class WorkerState(CopilotKitState):
    """State for individual tool chain execution workers. It is derived from CopilotKitState, which provides 'messages' list to store the conversation history"""
    sub_query: str
    additional_material: Optional[List[str]]
    tool_chain: List[str]
    tool_calls_history: List[ToolCall]
    tool_results_history: List[ToolMessage]
    worker_result: List[str] # To store the final result of the worker's execution, which will be aggregated into the parent agent's state
    last_tool_artifact: Optional[Any]
