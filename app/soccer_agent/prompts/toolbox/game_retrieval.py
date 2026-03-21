from langchain_core.prompts import ChatPromptTemplate

def get_game_info_retrieval_prompt_template() -> ChatPromptTemplate:
    prompt_template = """
    You are a soccer expert. Answer the question based ONLY on the provided match related information (metadata).

    User Question: "{query}"

    Match Information:
    {context}

    Please provide the answer based on the match related information. Make sure your answer is evidence-based and accurate.
    """
    return ChatPromptTemplate.from_template(prompt_template)

def get_game_history_retrieval_prompt_template() -> ChatPromptTemplate:
    prompt_template = """
    You are a soccer expert. Answer the question based ONLY on the provided match history (live commentary/annotations).
    
    User Question: "{query}"

    Match History (List of Annotations):
    {context}

    Please provide the answer based on the match history information. Think carefully about timestamps and event sequences. Make sure your answer is evidence-based and accurate.
    """
    return ChatPromptTemplate.from_template(prompt_template)