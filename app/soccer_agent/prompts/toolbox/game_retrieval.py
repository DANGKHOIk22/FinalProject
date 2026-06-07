from langchain_core.prompts import ChatPromptTemplate

def get_game_info_retrieval_prompt_template() -> ChatPromptTemplate:
    prompt_template = """
    You are a soccer expert. Answer the question based ONLY on the provided match related information (metadata).

    User Question: "{query}"

    Match Information:
    {context}

    Time Context: {time_context}

    Please provide the answer based on the match related information. Make sure your answer is evidence-based and accurate.
    MANDATORY: Your answer MUST be in ENGLISH.
    """
    return ChatPromptTemplate.from_template(prompt_template)

def get_game_history_retrieval_prompt_template() -> ChatPromptTemplate:
    prompt_template = """
    You are a soccer expert. Answer the question based ONLY on the provided match history (live commentary/annotations).
    
    User Question: "{query}"

    Match History (List of Annotations):
    {context}

    Time Context: {time_context}
    Current Playback Position: {video_position}

    If a Current Playback Position is given (not "None"), the user is watching this match live —
    focus your answer on events at or just before that moment. If "None", answer over the full history.

    Please provide the answer based on the match history information. Think carefully about timestamps and event sequences. Make sure your answer is evidence-based and accurate.
    MANDATORY: Your answer MUST be in ENGLISH.
    """
    return ChatPromptTemplate.from_template(prompt_template)