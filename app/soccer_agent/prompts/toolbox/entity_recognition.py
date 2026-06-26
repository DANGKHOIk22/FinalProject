from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


def get_entity_disambiguation_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a soccer player identification assistant. "
            "You will be given a cropped face image and a list of candidate player names. "
            "Your job is to identify which player in the list best matches the face. "
            "If you are not confident, respond with exactly 'unknown'."
        ),
        HumanMessagePromptTemplate.from_template(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{crop_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        "Candidate players: {candidate_names}\n\n"
                        "Respond with ONLY one exact name from the list above, "
                        "or 'unknown' if you cannot determine it confidently."
                    ),
                },
            ]
        ),
    ])
