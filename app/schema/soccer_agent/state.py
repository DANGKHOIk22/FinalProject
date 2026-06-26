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
class PlannedChain(BaseModel):
    """A single planned tool chain with per-chain confidence and ambiguity metadata."""
    chain: List[str] = Field(description="Ordered list of tool names to call for this sub-query")
    sub_query: str = Field(description="The specific sub-query this chain answers (MUST BE IN ENGLISH)")
    confidence: float = Field(description="Confidence score 0.0-1.0 for this chain being correct and answerable")
    is_ambiguous: bool = Field(description="True if this chain cannot be planned with sufficient confidence")
    clarifying_question: Optional[str] = Field(default=None, description="Question to ask the user when is_ambiguous=True (in user's language)")


class UnifiedPlanningOutput(BaseModel):
    """Structured output for combined query understanding and tool chain planning."""
    clarified_query: str = Field(description="User query with pronouns resolved, jargon expanded, and ASCII normalized")
    need_call_tools: bool = Field(description="False only for greetings or when the answer is already in conversation history")
    planned_chains: Optional[List[PlannedChain]] = Field(default=None, description="List of planned chains, one per parallel worker")


class GuardrailVerdict(BaseModel):
    """Structured verdict from the soccer-topic guardrail classifier."""
    is_soccer_related: bool = Field(description="True if the query is about soccer (players, teams, coaches, matches, leagues, statistics, or the live match the user is watching), or a follow-up within an ongoing soccer conversation. When uncertain, prefer True.")
    reason: str = Field(description="Brief justification for the verdict")

# Define the state structure for the agent
class AgentState(CopilotKitState):
    """Parent state structure for the planning agent. It is derived from CopilotKitState, which provides 'messages' list to store the conversation history"""
    clarified_query: str
    additional_material: Optional[Dict[str, Any]]  # {"game_id": str | None, "image_id": List[str], "video_id": str | None}
    planning_output: Optional[UnifiedPlanningOutput]
    tool_chains: Optional[List[List[str]]]       # non-ambiguous chains unpacked from planned_chains
    sub_queries: Optional[List[str]]             # sub-queries for non-ambiguous chains
    need_call_tools: Optional[bool]              # False = skip all workers
    pending_clarifications: Optional[List[str]]  # clarifying questions from ambiguous chains
    planning_error: Optional[str]                # set when planning output parsing fails — aggregator surfaces a system error
    parallel_results: Annotated[List[str], _reset_or_add]
    tool_calls_history: Annotated[List[ToolCall], _reset_or_add]
    tool_results_history: Annotated[List[ToolMessage], _reset_or_add]
    conversation_history: Optional[str]
    long_term_context: Optional[str]
    retrieved_cases: Optional[List[str]]
    time_context: Optional[str]
    video_current_time: Optional[float] # Current HLS video playback position in seconds (synced from frontend)
    is_off_topic: Optional[bool] # Set by the guardrail node: True = query is not soccer-related, hard-stop to refusal

class WorkerState(CopilotKitState):
    """State for individual tool chain execution workers. It is derived from CopilotKitState, which provides 'messages' list to store the conversation history"""
    worker_index: Optional[int]
    worker_id: Optional[str]
    sub_query: str
    additional_material: Optional[Dict[str, Any]]  # {"game_id": str | None, "image_id": List[str], "video_id": str | None}
    tool_chain: List[str]
    tool_calls_history: List[ToolCall]
    tool_results_history: List[ToolMessage]
    worker_result: List[str] # To store the final result of the worker's execution, which will be aggregated into the parent agent's state
    last_tool_artifact: Optional[Any]
    time_context: Optional[str] # Current date and time passed from AgentState
    video_current_time: Optional[float] # HLS playback position forwarded from AgentState
