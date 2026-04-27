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
