import logging
from langchain.tools import tool
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from toolbox.textual_entity_search import SearchingResult, SoccerEntities
from models.soccerwiki_entities import PlayerSchema, RefereeSchema, VenueSchema, TeamSchema
from prompts.toolbox.textual_retrieval_augment import retrieval_augment_prompt

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
        aggregated_text += f'SAU ĐÂY LÀ THÔNG TIN CỦA {entity.NAME}:\n'
        aggregated_text += f'(LOẠI THỰC THỂ: {entity.ENTITY_TYPE})\n'
        if entity.SUMMARY:
            aggregated_text += f'SUMMARY: {entity.SUMMARY}\n'
        if entity.INFOBOX:
            aggregated_text += f'INFOBOX: {entity.INFOBOX}\n'
        if entity.CONTENT:
            aggregated_text += f'CONTENT: {entity.CONTENT}\n'
        aggregated_text += '-' *10 + '\n\n'
    
    if searching_result.missing_entities:
        aggregated_text += 'KHÔNG TÌM THẤY THÔNG TIN CỦA CÁC THỰC THỂ SAU: '
        aggregated_text += ', '.join(searching_result.missing_entities) + '\n'

    return aggregated_text

@tool
def textual_retrieval_augment(query: str, searching_result: SearchingResult) -> str:
    """
    Given a text query, the tool retrieves the relevant information from given soccer information or database page. 
    It's always be used for background information of players, teams, coaches, referees, venues, etc.

    Args:
        query (str): Prompt query could be the original question, or the well defined question that can help retrieve the question.
        searching_result (SearchingResult): The searching result object containing information about found entities and missing entities from textual_entity_search tool.
    
    Returns:
        str: The final answer generated based on the retrieved information.
    """

    model = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest", 
        temperature=0.5,  
        top_p=0.95
    )

    retrieval_augment_chain = retrieval_augment_prompt | model

    # Prepare inputs
    searching_result_text = aggregate_searching_results(searching_result)
    inputs = {
        "user_query": user_query,
        "searching_result": searching_result_text
    }
    logger.info(f"Prepared inputs for textual retrieval augment tool: User Query: {inputs['user_query']}, Searching Result: {len(searching_result.found_entities)} found entities - {len(searching_result.missing_entities)} missing entities")
    logger.debug(f"Searching Result Text: {inputs['searching_result']}")
    logger.debug(f"Retrieval Augment Prompt: {retrieval_augment_prompt}")

    # Get answer
    try: 
        final_answer = retrieval_augment_chain.invoke(inputs)
        logger.info(f"Raw textual retrieval augment answer: {final_answer}")
        final_answer = final_answer.content
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        final_answer = "Sorry, I couldn't generate a response."


    return final_answer

if __name__ == "__main__":
    # Create logging config
    logging.basicConfig(level=logging.INFO)


    user_query = "Tell me about Lionel Messi's career highlights."

    # Mock searching result
    searching_result = SearchingResult(
        found_entities=[
            PlayerSchema(
                NAME="Lionel Messi",
                ENTITY_TYPE="Player",
                SUMMARY="Lionel Messi is an Argentine professional footballer who plays as a forward.",
                INFOBOX={"Position": "Forward", "Club": "Inter Miami", "Nationality": "Argentine"}
            )
        ]
    )

    # Call the function
    print(textual_retrieval_augment(user_query, searching_result))