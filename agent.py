from typing import TypedDict, Annotated, List, Optional, Callable
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import logging

from dotenv import load_dotenv
from prompts.agent import planning_prompt, execution_prompt
from config import (
    DEFAULT_MODEL, MODEL_TEMPERATURE, MODEL_TOP_P, MAX_COMPLETION_TOKENS,
    LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL
)
from toolbox.tools import (
    game_search, game_info_retrieval, match_history_retrieval,
    entity_recognition, textual_entity_search, textual_retrieval_augment,
    choice_selection, get_all_tools
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)


# Structured output models for LLM responses
class PlanningOutput(BaseModel):
    """Structured output for tool chain planning."""
    known_info: List[str] = Field(description="List of information items that are directly provided or known from the query")
    tool_chain: List[str] = Field(description="Ordered list of tools needed to answer the query")


# Define the state structure for the agent
class AgentState(TypedDict):
    """State structure for the planning agent."""
    user_query: str
    additional_material: Optional[str]
    known_info: List[str]
    tool_chain: List[str]
    execution_history: List[dict]
    step_results: List[str]


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
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=MODEL_TEMPERATURE, 
            top_p=MODEL_TOP_P,
            max_output_tokens=MAX_COMPLETION_TOKENS
        )
        self.planning_parser = PydanticOutputParser(pydantic_object=PlanningOutput)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Use LangChain tools from tools.py
        self.tools = get_all_tools()
        
        # Tool mapping dictionary using LangChain @tool decorated functions
        self.tool_registry: dict[str, Callable] = {
            "game_search": game_search,
            "game_info_retrieval": game_info_retrieval,
            "match_history_retrieval": match_history_retrieval,
            "entity_recognition": entity_recognition,
            "textual_entity_search": textual_entity_search,
            "textual_retrieval_augment": textual_retrieval_augment,
            "choice_selection": choice_selection,
        }
        
        self.graph = self._build_graph()
        self.logger.info(f"SoccerAgent initialized with model: {model_name}")
        self.logger.info(f"Loaded {len(self.tools)} LangChain tools")
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("tool_chain_planning", self._tool_chain_planning)
        workflow.add_node("iterative_tool_execution", self._iterative_tool_execution)
        
        # Define the flow
        workflow.set_entry_point("tool_chain_planning")
        workflow.add_edge("tool_chain_planning", "iterative_tool_execution")
        workflow.add_conditional_edges(
            "iterative_tool_execution",
            self.should_continue_or_end,
            {
                "end_plan_complete": END, 
                "continue_execution": "iterative_tool_execution" 
            }
        )
        
        return workflow.compile()
    
    def _tool_chain_planning(self, state: AgentState) -> AgentState:
        """
        Combined node: Prepare prompt, analyze query, and parse to generate tool chain.
        Uses structured output for reliable parsing.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with parsed known_info and tool_chain
        """
        # Format additional material
        additional_material = state.get("additional_material", "None")
        if not additional_material or additional_material.strip() == "":
            additional_material = "None"
        
        # Build tool descriptions
        tool_descriptions = ""
        for tool in self.tools:
            tool_descriptions += f"- {tool.name}: {tool.description}\n"
        
        # Get format instructions
        format_instructions = self.planning_parser.get_format_instructions()
        
        # Create formatted prompt by replacing placeholders in the original template
        from copy import deepcopy
        formatted_prompt = deepcopy(planning_prompt)
        
        # Replace placeholders in HumanMessage content
        human_msg_content = formatted_prompt[1].content
        human_msg_content = human_msg_content.replace("{{toolbox_descriptions}}", tool_descriptions)
        human_msg_content = human_msg_content.replace("{{user_query}}", state["user_query"])
        human_msg_content = human_msg_content.replace("{{additional_material}}", additional_material)
        human_msg_content += f"\n\n{format_instructions}"
        
        # Create new HumanMessage with formatted content
        from langchain_core.messages import HumanMessage
        formatted_prompt[1] = HumanMessage(content=human_msg_content)
        
        # Get LLM response
        response = self.llm.invoke(formatted_prompt)
        response_text = response.content
        
        # Google Gemini models may not have reasoning_content in additional_kwargs
        if hasattr(response, 'additional_kwargs') and 'reasoning_content' in response.additional_kwargs:
            self.logger.info(f"LLM reasoning: {response.additional_kwargs['reasoning_content']}")
        
        # Parse with PydanticOutputParser
        planning_output: PlanningOutput = self.planning_parser.parse(response_text)
        
        self.logger.info("="*70)
        self.logger.info("STRUCTURED PLANNING OUTPUT")
        self.logger.info("="*70)
        
        self.logger.info(f"Known Info: {planning_output.known_info}")
        self.logger.info(f"Tool Chain: {planning_output.tool_chain}")
        self.logger.info("="*70)
     
        # Use structured output directly
        state["known_info"] = planning_output.known_info
        state["tool_chain"] = planning_output.tool_chain
       
        return state
    
    def _iterative_tool_execution(self, state: AgentState) -> AgentState:
        """
        Iteratively execute the tool chain step by step using bind_tools with tool_choice.
        This allows the LLM to see the full tool schema and automatically generate correct parameters.
        
        Args:
            state: Current agent state with known_info and tool_chain
            
        Returns:
            Updated state with execution results
        """
        query = state["user_query"]
        additional_material = state.get("additional_material", "None")
        known_info = state["known_info"]
        tool_chain = state["tool_chain"]
        execution_history = state.get("execution_history", [])
        step_results = state.get("step_results", [])
        
        self.logger.info("="*70)
        self.logger.info("🔧 ITERATIVE TOOL EXECUTION - CURRENT STEP")
        self.logger.info("="*70)
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Known Info: {known_info}")
        self.logger.info(f"Steps Completed: {len(execution_history)}")
        self.logger.info("="*70)
        
        # Get current tool from chain
        current_tool_name = tool_chain[0]
        current_tool = self.tool_registry.get(current_tool_name)
        self.logger.info(f"Current Tool Name: {current_tool_name}")
        if not current_tool:
            error_msg = f"Tool '{current_tool_name}' not found in registry"
            self.logger.error(error_msg)
            state["step_results"].append(error_msg)
            return state
        
        self.logger.info(f"🎯 Executing Tool: {current_tool_name}")
        self.logger.info(f"📝 Tool Description: {current_tool.description}")
        self.logger.info(f"🔧 Tool Function Name: {current_tool.name}")
    
        model_with_tool = self.llm.bind_tools([current_tool])
        
        # Build execution history string
        history_str = self._build_history_string(state, execution_history, step_results)
        
        # Create formatted execution prompt
        from copy import deepcopy
        from langchain_core.messages import HumanMessage
        formatted_exec_prompt = deepcopy(execution_prompt)
        
        # Replace placeholders in HumanMessage content
        exec_msg_content = formatted_exec_prompt[1].content
        exec_msg_content = exec_msg_content.replace("{{user_query}}", state["user_query"])
        exec_msg_content = exec_msg_content.replace("{{additional_material}}", str(additional_material))
        exec_msg_content = exec_msg_content.replace("{{known_info}}", str(known_info))
        exec_msg_content = exec_msg_content.replace("{{tool_chain}}", " -> ".join(tool_chain))
        exec_msg_content = exec_msg_content.replace("{{history}}", history_str)
        
        # Create new HumanMessage with formatted content
        formatted_exec_prompt[1] = HumanMessage(content=exec_msg_content)
        
        # Get LLM response with tool calling
        try:
            response = model_with_tool.invoke(formatted_exec_prompt)

            self.logger.info(f"🤖 LLM Response: {response}")

            # Check if tool was called
            if not response.tool_calls or len(response.tool_calls) == 0:
                error_msg = f"LLM did not generate a tool call for {current_tool_name}"
                self.logger.error(error_msg)
                self.logger.error(f"Response content: {response.content}")
                
                step_results.append(error_msg)
                execution_history.append({
                    "tool": current_tool_name,
                    "query": "ERROR - No tool call generated",
                    "material": "ERROR",
                })
            else:
                # Extract the tool call
                tool_call = response.tool_calls[0]
                
                self.logger.info("="*70)
                self.logger.info(f"📞 Tool Call Generated:")
                self.logger.info(f"   Tool Name: {tool_call['name']}")
                self.logger.info(f"   Tool Args: {tool_call['args']}")
                self.logger.info("="*70)
                
                # Extract query and material from args
                tool_query = tool_call['args'].get('query', '')
                tool_material = tool_call['args'].get('material', 'None')
                
                self.logger.info(f"❓ Query: {tool_query}")
                self.logger.info(f"📦 Material: {tool_material}")
                
                # Execute the tool using LangChain's invoke method
                tool_result = current_tool.invoke(tool_call['args'])
                
                self.logger.info(f"✅ Tool Result: {tool_result}")
                
                # Store execution history
                execution_history.append({
                    "tool": current_tool_name,
                    "query": tool_query,
                    "material": tool_material,
                })
                
                step_results.append(tool_result)
            
        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.error(traceback.format_exc())
            
            step_results.append(error_msg)
            execution_history.append({
                "tool": current_tool_name,
                "query": "ERROR",
                "material": "ERROR",
            })
        
        # Update state
        tool_chain = tool_chain[1:]  # Remove the executed tool
        state["execution_history"] = execution_history
        state["step_results"] = step_results
        state["tool_chain"] = tool_chain
   
        
        self.logger.info("="*70)
        self.logger.info("✅ TOOL EXECUTION STEP COMPLETED")
        self.logger.info(f"   Steps completed: {len(execution_history)}")
        self.logger.info(f"   Remaining tools: {len(tool_chain)}")
        self.logger.info("="*70)
        
        return state
    def should_continue_or_end(self,state: AgentState):
        if not state.get("tool_chain"):
            # Kế hoạch đã chạy hết -> Dừng
            return "end_plan_complete"
        else:
            # Vẫn còn kế hoạch -> Tiếp tục lặp
            return "continue_execution"
    
    def _build_history_string(self,state:AgentState, execution_history: List[dict], step_results: List[str]) -> str:
        """
        Build the execution history string for the prompt.
        
        Args:
            execution_history: List of executed steps
            step_results: List of results from each step
            
        Returns:
            Formatted history string
        """
        if not execution_history:
            return "No execution history yet. This is the first step."
        
        history_parts = []
        for idx, (exec_step, result) in enumerate(zip(execution_history, step_results)):
            call_type = "EndCall" if not state.get("tool_chain") else "Call"
            history_parts.append(f"""
            <{call_type}>
            <Query>{exec_step['query']}</Query>
            <Material>{exec_step['material']}</Material>
            <Tool>{exec_step['tool']}</Tool>
            </{call_type}>

            <StepResult>
            <Answer>{result}</Answer>
            </StepResult>
            """)
        
        return "\n".join(history_parts)
    

    def run(self, user_query: str, additional_material: Optional[str] = None) -> dict:
        """
        Run the complete workflow: planning + iterative tool execution.
        
        Args:
            user_query: The user's question about soccer
            additional_material: Optional additional context (e.g., image paths)
            
        Returns:
            Dictionary containing complete results from planning and execution
        """
        self.logger.info(f"Starting run for query: {user_query[:100]}...")
        
        # Initialize state
        initial_state = {
            "user_query": user_query,
            "additional_material": additional_material if additional_material else "None",
            "known_info": [],
            "tool_chain": [],
            "execution_history": [],
            "step_results": [],
        }
        
        # Run the full graph (planning → execution)
        final_state = self.graph.invoke(initial_state)
        
        # Prepare the complete results
        result = {
            "user_query": user_query,
            "known_info": final_state["known_info"],
            "tool_chain": final_state["tool_chain"],
            "planning_raw_response": final_state.get("planning_raw_response", ""),
            "execution_history": final_state["execution_history"],
            "execution_raw_responses": final_state.get("execution_raw_responses", []),
            "step_results": final_state["step_results"],
        }
        
        self.logger.info("Run completed successfully")
        return result
    

def main():
    """Example usage of the Soccer Planning Agent."""
    
    # Initialize the agent with Google Gemini model
    agent = SoccerAgent(model_name=DEFAULT_MODEL)
    
    # # Example 10:
    # logger.info("="*60)
    # logger.info("EXAMPLE 10")
    # logger.info("="*60)
    # query10 = "Which match had more total fouls: the 2018 Champions League final or the 2022 Champions League final?"
    # agent.run(query10)
    
    # # Example 11:
    # logger.info("="*60)
    # logger.info("EXAMPLE 11")
    # logger.info("="*60)
    # query11 = "Who scored the goal in the 2014 Champions League final, and what was the first professional club he ever played for?"
    # agent.run(query11)

    # Example 12:
    logger.info("="*60)
    logger.info("EXAMPLE 12")
    logger.info("="*60)
    query12 = "For the player in the image, in what stadium did he make his UEFA Champions League debut?"
    additional_material12 = 'image": ["/path/to/player.jpg"]'
    agent.run(query12, additional_material12)
    


if __name__ == "__main__":
    main()
