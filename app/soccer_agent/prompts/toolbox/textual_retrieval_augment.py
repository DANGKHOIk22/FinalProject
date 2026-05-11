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

            If the question is not related to soccer/football, state that you can only answer questions related to soccer/football topics.

            ### REQUIREMENTS
            The answer **must be strictly based** on the information provided in the **SEARCHING RESULT**. Do not add, omit, or infer any information.

            ### OUTPUT FORMAT INSTRUCTIONS
            You must provide your response in a structured format with the following fields:
            1. **answer**: The final synthesized answer based on the provided search results.
            2. **has_sufficient_info**: Set this to `true` if the search results provided enough information to fully and accurately answer the user's query. Set to `false` if the information is missing, incomplete, or if you had to rely on web fallback indicators.
            3. **entity_types**: If the search results indicate that an entity is missing from the database (e.g., "NOT FOUND INFORMATION FOR..."), or if the context suggests the entity is new, you MUST classify it. Map the entity name to one of: 'player', 'team', 'venue', 'referee'. If all entities are found and accounted for, leave this dictionary empty.

            ### USER QUERY
            {query}

            ### SEARCHING RESULT
            {searching_result}"""
        )
    ])
    return retrieval_augment_prompt_template