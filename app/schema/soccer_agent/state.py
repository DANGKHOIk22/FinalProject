from typing import Annotated, Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import ToolMessage, ToolCall
from copilotkit import CopilotKitState


def _reset_or_add(left: list, right: list) -> list:
    """Reducer for parallel-aggregated lists: empty right = reset, non-empty = add.
    This lets initial_state use [] to clear stale checkpoint data between turns,
    while still allowing concurrent workers to accumulate results via Send."""
    if not right:
        return []
    return (left or []) + right

# Structured output models for LLM responses
class UnifiedPlanningOutput(BaseModel):
    """Structured output for combined query understanding and tool chain planning."""
    clarified_query: str = Field(description="User query with pronouns resolved, jargon expanded, and ASCII normalized")
    is_ambiguous: bool = Field(description="True if the query cannot be resolved unambiguously")
    clarifying_questions: List[str] = Field(description="Questions to ask the user when is_ambiguous=True")
    tool_chains: Optional[List[List[str]]] = Field(description="List of independent tool chains to answer the query")
    sub_queries: Optional[List[str]] = Field(description="List of specific decomposed sub-queries, each corresponding to a tool chain")
    need_call_tools: Optional[bool] = Field(description="Indicates whether tool calls are necessary")

# Define the state structure for the agent
class AgentState(CopilotKitState):
    """Parent state structure for the planning agent. It is derived from CopilotKitState, which provides 'messages' list to store the conversation history"""
    clarified_query: str # User query with pronouns/abbreviations resolved by QueryUnderstandingPipeline
    additional_material: Optional[List[str]] # Additional material (e.g, image/video related to the user question)
    planning_output: Optional[UnifiedPlanningOutput] # The output from the planning step, which includes tool chains and sub-queries
    parallel_results: Annotated[List[str], _reset_or_add] # Aggregated parallel results
    tool_calls_history: Annotated[List[ToolCall], _reset_or_add] # History of tool calls
    tool_results_history: Annotated[List[ToolMessage], _reset_or_add] # History of tool results
    conversation_history: Optional[str] # Optional conversation history for context
    long_term_context: Optional[str] # Context retrieved from pgvector long-term memory
    retrieved_cases: Optional[List[str]] # Few-shot planning examples retrieved from case bank
    recent_msgs_for_qu: Optional[List[Any]] # Passed from history to understanding
    effective_memory: Optional[Any] # Passed from history to understanding
    time_context: Optional[str] # Current date and time for temporal reasoning

class WorkerState(CopilotKitState):
    """State for individual tool chain execution workers. It is derived from CopilotKitState, which provides 'messages' list to store the conversation history"""
    sub_query: str
    additional_material: Optional[List[str]]
    tool_chain: List[str]
    tool_calls_history: List[ToolCall]
    tool_results_history: List[ToolMessage]
    worker_result: List[str] # To store the final result of the worker's execution, which will be aggregated into the parent agent's state
    last_tool_artifact: Optional[Any]
    time_context: Optional[str] # Current date and time passed from AgentState
