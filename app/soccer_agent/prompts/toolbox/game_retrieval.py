from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


def get_game_info_retrieval_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a soccer expert. "
            "Answer the user's question based ONLY on the match metadata provided — "
            "never use your internal knowledge. "
            "Be evidence-based and accurate. "
            "Respond in ENGLISH only."
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: {query}\n\n"
            "Match Information:\n{context}\n\n"
            "Time Context: {time_context}"
        ),
    ])


def get_game_history_retrieval_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a soccer expert. "
            "Answer the user's question based ONLY on the provided match history "
            "(live commentary or annotations) — never use your internal knowledge. "
            "If Current Playback Position is not 'None', the user is watching the match live — "
            "focus your answer on events at or just before that moment. "
            "If it is 'None', answer over the full match history. "
            "Think carefully about timestamps and event sequences. "
            "Be evidence-based and accurate. "
            "Respond in ENGLISH only."
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: {query}\n\n"
            "Match History (List of Annotations):\n{context}\n\n"
            "Time Context: {time_context}\n"
            "Current Playback Position: {video_position}"
        ),
    ])
