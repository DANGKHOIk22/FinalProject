from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


def get_commentary_generation_prompt_template() -> ChatPromptTemplate:
    """Prompt template for generating soccer match commentary + structured annotations from video."""

    return ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an expert soccer match commentator and analyst.\n\n"
            "You will receive one or more short video clips. First determine whether they depict a soccer match.\n\n"
            "If the clips do NOT show a soccer match: clearly state that, return an empty annotations list, and stop.\n\n"
            "If the clips DO show a soccer match, produce TWO outputs:\n"
            "1. A natural-language match commentary (~500 words, concise, not repetitive, IN ENGLISH) "
            "describing what happens in the provided clips.\n"
            "2. A list of structured key-event annotations grounded in the video content "
            "(do not invent events not visible in the clips).\n\n"
            "If you are not confident the clips depict a soccer match, set the annotations list to empty."
        ),
        HumanMessagePromptTemplate.from_template(
            [
                {
                    "type": "text",
                    "text": "{query_context}### OUTPUT FORMAT\n{output_format}\n"
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "{video_url}"}
                }
            ]
        ),
    ])
