import logging
from typing import Annotated, Literal, Tuple, Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool, InjectedState
from langchain_core.callbacks import CallbackManagerForToolRun

from app.config.config import DEFAULT_MODEL
from app.toolbox.textual_entity_search import SearchingResult
from app.prompts.toolbox.textual_retrieval_augment import get_textual_retrieval_augment_prompt_template

logger = logging.getLogger(__name__)



class TextualRetrievalAugmentInput(BaseModel):
    query: str = Field(description="Prompt query could be the original question, or the well defined question that can help retrieve the question.")
    execution_agent_state: Annotated[dict, InjectedState] = Field(description="The state of the execution agent, containing artifacts from previous tools.")

class TextualRetrievalAugmentTool(BaseTool):
    name: str = "textual_retrieval_augment"
    description: str = """
    Given a text query, the tool retrieves the relevant information from given soccer information or database page. 
    It's always be used for background information of players, teams, coaches, referees, venues, etc. The data from previous tool call will be retrieved automatically.
    """
    args_schema:Type[BaseModel] = TextualRetrievalAugmentInput # type: ignore
    response_format: Literal['content', 'content_and_artifact'] = 'content_and_artifact'
    
    _llm: ChatGoogleGenerativeAI = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._llm = ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL, 
            temperature=0.5,  
            top_p=0.95
        )

    def _run(self, query: str, execution_agent_state: Annotated[dict, InjectedState], run_manager: Optional[CallbackManagerForToolRun] = None) -> Tuple[str, None]:
        """
        Given a text query, the tool retrieves the relevant information from given soccer information or database page. 
        It's always be used for background information of players, teams, coaches, referees, venues, etc. The data from previous tool call will be retrieved automatically.
        Args:
            query (str): Prompt query could be the original question, or the well defined question that can help retrieve the question.
            
        Returns:
            Tuple[str, None]:
                str: The final answer generated based on the retrieved information.
                None: There is no return artifact for this tool.
        """ 

        # Build the retrieval augment chain
        retrieval_augment_prompt_template = get_textual_retrieval_augment_prompt_template()
        retrieval_augment_chain = retrieval_augment_prompt_template | self._llm

        # Get the searching result from the execution agent state then aggregate it
        searching_result: SearchingResult = execution_agent_state.get('last_tool_artifact', None) # type: ignore
        if not searching_result:
            logger.warning("No artifact found in execution agent state for textual retrieval augment tool.")
            return "Could not retrieve any information from previous tool calls. Please ensure that the previous tools have been executed successfully or try calling the previous tools again.", None
        else: 
            logger.info(f"Artifact from execution agent state retrieved for textual retrieval augment tool: {searching_result}.")
        
        searching_result_text = self._aggregate_searching_results(searching_result)
        
        # Create inputs for retrieval augment chain
        inputs = {
            "query": query,
            "searching_result": searching_result_text
        }
        logger.info(f"Prepared inputs for textual retrieval augment tool: User Query: {inputs['query']}, Searching Result: {len(searching_result.found_entities)} found entities - {len(searching_result.missing_entities)} missing entities")
        logger.debug(f"Searching Result Text: {inputs['searching_result']}")
        logger.debug(f"Retrieval Augment Prompt: {retrieval_augment_prompt_template}")

        # Get answer
        try: 
            final_answer = retrieval_augment_chain.invoke(inputs)
            logger.info(f"Raw textual retrieval augment answer: {final_answer}")
            return str(final_answer.content), None
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return "An error occurred while generating the answer based on the retrieved information. Try calling this tool again or stop the execution.", None
    
    @staticmethod
    def _aggregate_searching_results(searching_result: SearchingResult) -> str:
        """
        Aggregate searching results into a textual format

        Args:
            searching_result (SearchingResult): The searching result object

        Returns:
            str: The aggregated searching results in textual format
        """
        
        aggregated_text = ""
        for entity in searching_result.found_entities:
            aggregated_text += '-' *10 + '\n'
            aggregated_text += f'INFORMATION ABOUT:  {entity.NAME}:\n'
            aggregated_text += f'(ENTITY TYPE: {entity.ENTITY_TYPE})\n'
            if entity.SUMMARY:
                aggregated_text += f'SUMMARY: {entity.SUMMARY}\n'
            if entity.INFOBOX:
                aggregated_text += f'INFOBOX: {entity.INFOBOX}\n'
            if entity.CONTENT:
                aggregated_text += f'CONTENT: {entity.CONTENT}\n'
            aggregated_text += '-' *10 + '\n\n'
        
        if searching_result.missing_entities:
            aggregated_text += 'NOT FOUND INFORMATION FOR THE FOLLOWING ENTITIES: '
            aggregated_text += ', '.join(searching_result.missing_entities) + '\n'

        return aggregated_text
