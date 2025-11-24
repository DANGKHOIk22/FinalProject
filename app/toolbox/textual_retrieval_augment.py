import logging
from typing import Annotated

from langchain.tools import tool
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage

from app.toolbox.textual_entity_search import SearchingResult
from app.prompts.toolbox.textual_retrieval_augment import get_textual_retrieval_augment_prompt_template

logger = logging.getLogger(__name__)

def aggregate_searching_results(searching_result: SearchingResult) -> str:
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

@tool
def textual_retrieval_augment(query: str, execution_agent_state: Annotated[dict, InjectedState]) -> str:
    """
    Given a text query, the tool retrieves the relevant information from given soccer information or database page. 
    It's always be used for background information of players, teams, coaches, referees, venues, etc. The data from previous tool call will be retrieved automatically.
    Args:
        query (str): Prompt query could be the original question, or the well defined question that can help retrieve the question.
        
    Returns:
        str: The final answer generated based on the retrieved information.
    """ 

    # Build the retrieval augment chain
    model = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest", 
        temperature=0.5,  
        top_p=0.95
    )

    retrieval_augment_prompt_template = get_textual_retrieval_augment_prompt_template()
    retrieval_augment_chain = retrieval_augment_prompt_template | model

    # Get the searching result from the execution agent state then aggregate it
    searching_result: SearchingResult = execution_agent_state.get('last_tool_artifact', None) # type: ignore
    logger.info("artifact from execution agent state retrieved for textual retrieval augment tool.",searching_result)
    searching_result_text = aggregate_searching_results(searching_result)
    
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
        final_answer.content
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        final_answer = "I'm sorry, but I couldn't generate an answer based on the retrieved information."

    return final_answer
