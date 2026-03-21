from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage


def get_commentary_generation_prompt_template() -> ChatPromptTemplate:
    """Prompt template for generating soccer match commentary + structured annotations from video."""

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "You are an expert soccer match commentator and analyst. "
                "You will be given one or more short video clips that may contain a soccer match. "
                "First, determine if the clips depict a soccer match. "
                "If they do not, clearly say so and return an empty list of annotations. "
                "If they do, write an engaging match commentary and extract key events."
            ),
            HumanMessagePromptTemplate.from_template(
                [
                    {
                        "type": "text",
                        "text": """### TASK
Generate:
1) A natural-language commentary (~500 words, concise, not repetitive) about what happens in the provided soccer video clips.
2) A list of structured event annotations.

### IMPORTANT
- If the video clips are not soccer-related or you are not confident they depict a soccer match, set annotations to an empty list.
- The annotations must be grounded in the video content.

### OUTPUT FORMAT
{output_format}
"""
                    },
                    {
                        "type": "media",
                        "source_type": "base64",
                        "mime_type": "{mime_type}",
                        "data": "{video_base64}",
                    }
                ]
                
            ),
        ]
    )

    
