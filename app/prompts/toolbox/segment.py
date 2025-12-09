from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
def get_segment_prompt_template() -> ChatPromptTemplate:
    prompt_template =ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a Soccer Domain Query Parser. Your task is to identify distinct entities (players, referees, teams,...) in a natural language query and segment them into independent descriptions."
        ),
        HumanMessagePromptTemplate.from_template(
        """
        ## INPUT FORMAT:
        Query: {query}
        
        ## OUTPUT FORMAT:
        {output_format}
        
        Return ONLY a valid JSON object with a "segments" key containing a list of strings. Each string is a segmented entity description.
        
        ### OBJECTIVE
        1.  **Identify Soccer Entities:** Look for specific entity types: Players, Goalkeepers, Referees, Managers, or specific names (e.g., "Messi", "Ronaldo").
        2.  **Distinguish Attributes vs. Entities:**
            * **Keep Together:** Attributes like jersey numbers, kit colors, teams, or accessories (e.g., "Player in red with number 7").
            * **Split:** When a new distinct entity is introduced (e.g., "Player A... AND Player B...").
        3.  **Standardize:** Ensure each split segment is a complete, descriptive string usable for image retrieval.

        ### RULES
        * **Jersey/Kit Logic:** "A player wearing a red and blue jersey" is ONE entity (Barcelona style). Do not split based on multiple colors on a single kit.
        * **Action Logic:** If two players are interacting but described distinctly, split them. (e.g., "A striker shooting and a goalkeeper saving" -> Split).
        * **Implicit Subjects:** If the user implies a second entity, make the split explicit.
        * **Output Format:** Return a list of segmented entity descriptions.

        ### FEW-SHOT EXAMPLES

        **Input:** "A player in a red jersey, a goalkeeper in a yellow kit"
        **Reasoning:** Two distinct entities with different roles and kit colors.
        **Output:** {{"segments": ["A player in a red jersey", "a goalkeeper in a yellow kit"]}}

        **Input:** "Messi wearing the number 10 Barcelona jersey"
        **Reasoning:** "Number 10" and "Barcelona jersey" are attributes of the single entity "Messi".
        **Output:** {{"segments": ["Messi wearing the number 10 Barcelona jersey"]}}
        
        **Input:** "Ronaldo dribbling and a defender chasing him"
        **Reasoning:** Distinct entities involved in an action sequence.
        **Output:** {{"segments": ["Ronaldo dribbling", "a defender chasing him"]}}

        **Input:** "The referee showing a red card to a player in blue"
        **Reasoning:** The referee and the player are separate entities.
        **Output:** {{"segments": ["The referee showing a red card", "a player in blue"]}}
        
        **Input:** "A team in white stripes and black shorts"
        **Reasoning:** This describes a single group entity/uniform style. Do not split colors.
        **Output:** {{"segments": ["A team in white stripes and black shorts"]}}
        
        Now process the query and return your answer in the exact JSON format shown above.
        """
        )
    ])
    
    return prompt_template