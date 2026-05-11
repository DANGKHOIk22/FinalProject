from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage


def get_textual_retrieval_augment_prompt_template() -> ChatPromptTemplate:
    """Create a prompt template for synthesizing information from retrieved soccer-related texts.
    It is used in the 'textual_retrieval_augment' tool."""

    retrieval_augment_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a helpful assistant that answers user requests on soccer/football topics by using the information provided. Ensure your answers are accurate and relevant."
        ),
        HumanMessagePromptTemplate.from_template(
            """### Context
            After synthesizing the information related to the entities identified in the user's query, your role is to compile that raw information into a complete and accurate final answer.

            ### INPUT FORMAT
            The SEARCHING RESULT will be provided as a structured list of text snippets/JSON objects containing the relevant facts.

            ### OUTPUT FORMAT
            Return a JSON object with exactly two fields:
            - "answer": the complete answer based solely on the SEARCHING RESULT. If data is insufficient, explain what is missing.
            - "has_sufficient_info": true ONLY if every entity was found AND the data fully answers the query. Set false if any entity is listed as NOT FOUND, if key facts are missing, or if you are uncertain.

            If the question is not related to soccer/football, set answer to 'I can only answer questions related to soccer/football topics.' and has_sufficient_info to true.

            ### REQUIREMENTS
            The answer **must be strictly based** on the information provided in the **SEARCHING RESULT**. Do not add, omit, or infer any information.

            ### USER QUERY
            {query}

            ### SEARCHING RESULT
            {searching_result}"""
        )
    ])
    return retrieval_augment_prompt_template