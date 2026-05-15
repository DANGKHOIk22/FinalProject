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
            The SEARCHING RESULT will be provided as a structured list of text snippets/JSON objects containing the relevant facts. Each entity has a `LAST_UPDATED` field.

            ### CRITICAL REQUIREMENTS (TEMPORAL INTEGRITY)
            1. **Strict Grounding**: The answer **must be strictly based** on the information provided in the **SEARCHING RESULT**.
            2. **STALENESS CHECK (DO NOT IGNORE)**:
                - Compare the `LAST_UPDATED` of each entity with the current `TIME CONTEXT`.
                - **STALE DATA**: For dynamic topics (current club, manager, injury, status, recent performance), if `LAST_UPDATED` is older than **6 months** relative to `TIME CONTEXT`, the data is considered **UNRELIABLE**.
                - **Example**: If `TIME CONTEXT` is in 2026 and `LAST_UPDATED` is from 2024, the data is **STALE**.
            3. **FALLBACK TRIGGER**: If the user query asks for "current", "latest", "now", or implies a present-day fact, and the available data is **STALE** (older than 6 months), you **MUST** set `has_sufficient_info: false`.
                - **DO NOT** answer with 2-year-old information if the user asks for the current situation.
                - **DO NOT** assume nothing has changed since the last update.
            4. **Incomplete Data**: If an entity is NOT FOUND or info is missing, set `has_sufficient_info: false`.
            5. **LANGUAGE**: Respond **ONLY in ENGLISH**, regardless of the user's input language.

            ### OUTPUT FORMAT INSTRUCTIONS
            You must provide your response in a structured format with the following fields:
            1. **answer**: The final synthesized answer (IN ENGLISH). If `has_sufficient_info` is false, this can be empty or a partial answer with a note that more recent data is needed.
            2. **has_sufficient_info**: Set to `true` ONLY if the data is present AND fresh (within 2 weeks for dynamic queries). Set to `false` if the data is missing, incomplete, or STALE (e.g. 2024 data vs 2026 context).
            3. **unknown_entities**: A list of objects for entities that are missing or need classification. Each object must have:
                - `entity_name`: The name of the entity.
                - `entity_type`: One of: 'player', 'team', 'venue', 'referee', or 'unknown'.

            ### USER QUERY
            {query}

            ### TIME CONTEXT
            {time_context}

            ### SEARCHING RESULT
            {searching_result}"""
        )
    ])
    return retrieval_augment_prompt_template