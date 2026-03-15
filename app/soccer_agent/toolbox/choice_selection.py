import logging
from typing import Any, Type, Literal, Optional, Annotated, Tuple
from pydantic import BaseModel, PrivateAttr, Field
from app.soccer_agent.prompts.toolbox.choice_selection import get_choice_selection_prompt_template
from app.config.config import DEFAULT_MODEL
from app.config.settings import Settings

from langchain.tools import BaseTool, InjectedState
from langchain.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_qwq import ChatQwQ
from langsmith import get_current_run_tree

logger = logging.getLogger(__name__)

class ChoiceSelectionInput(BaseModel):
    query: str = Field(description="A query containing the question and its according options (in forms of 'o1', 'o2', ......)")
    open_ended_answer: str = Field(description="The open-ended answer generated earlier for the question in the query")
    execution_agent_state: Annotated[dict, InjectedState] = Field(description="The state of the execution agent, containing artifacts from previous tools.")

class ChoiceSelection(BaseTool):
    name: str = "choice_selection"
    description: str = "Given an open-ended answer to a question, the tool identifies the most appropriate answer choice from a set of of closed-ended (multiple-choice) options. It analyzes the open answer and matches it to the correct option. So this tool is used at the final step of question answering when the question is in multiple-choice format. The input to this tool includes the question with its options and the open-ended answer generated earlier. The output is the selected choice (e.g., 'o1', 'o2', etc.) that best corresponds to the open answer."
    args_schema: Type[BaseModel] = ChoiceSelectionInput # type: ignore
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    _llm: BaseChatModel = PrivateAttr()

    def __init__(self, llm: Optional[BaseChatModel] = None):
        super().__init__()
        self._llm = llm or ChatQwQ(
            model=DEFAULT_MODEL, 
            temperature=0.5,  
            top_p=0.95,
            api_key=Settings.DASHSCOPE_API_KEY
        )

    def _run(self, query: str, open_ended_answer: str, execution_agent_state: Annotated[dict, InjectedState], run_manager: Optional[CallbackManagerForToolRun]) -> Tuple[str, None]:
        """
        Execute the choice selection tool to identify the most appropriate answer choice from multiple-choice options.
        
        Args:
            query: A query containing the question and its options (in forms of 'o1', 'o2', ...)
            open_ended_answer: The open-ended answer generated earlier for the question
            execution_agent_state: The state of the execution agent containing artifacts from previous tools
            run_manager: Optional callback manager for tool run
            
        Returns:
            A tuple containing the selected choice (e.g., 'o1', 'o2', etc.) and None
        """

        # Create the prompt by using the choice selection prompt template
        prompt_template = get_choice_selection_prompt_template()
        formatted_prompt = prompt_template.format_messages(
            query=query,
            open_ended_answer=open_ended_answer
        )

        try:
            # Invoke LLM to get the selected choice
            response = self._llm.invoke(formatted_prompt)
            
            # Extract the choice from the response
            selected_choice = str(response.content).strip() if hasattr(response, 'content') else str(response).strip()
            
            logger.info(f"Selected choice: {selected_choice}")
            
            return selected_choice, None
            
        except Exception as e:
            error_msg = f"Error in choice selection tool: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Send error to LangSmith run tree
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.end(
                    error=error_msg
                )
            
            return "An error occurred while selecting the choice. Try again or stop the execution.", None
