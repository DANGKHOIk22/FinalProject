from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


# Create system and user messages
def get_planning_prompt_template() -> ChatPromptTemplate:
    planning_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are a multi-modal agent that can answer questions about soccer knowledge."
        ),
        HumanMessagePromptTemplate.from_template(
            """For each question, you will receive:
- A question about soccer considering different aspects of soccer
- You might also receive one or more video clips or images as context
Your task involves three sequential parts:
1. Problem Decomposition (Part 1)
- Identify available information
- Break down the question into sequential steps
2. Sequential Tool Application (Part 2)
- Execute one tool at a time
- Record each tool’s output
- Continue until sufficient information is gathered
3. Solution Synthesis (Part 3)
- Integrate all results
- Generate final answer

## Available Tools
For all the QA, you need to decompose them and Here are the tools that you can use to answer the questions:
{toolbox_descriptions}

## Response Instructions
You must respond with a plan that populates the following two fields based on your analysis. The framework will handle formatting.
1.  **known_info**: A list of information categories explicitly mentioned in the query and material (e.g., $GameContext$, $PlayerContext$, $Image$).
2.  **tool_chain**: A list of EXACT tool names (e.g., "game_search", "game_info_retrieval") needed to answer the query, in the order they should be executed.

## Output Format Instructions
Follow these instructions carefully to ensure your response is correctly formatted:
{format_instructions}
## Examples
* **Purpose:** These examples teach you *how to reason* to determine the correct `known_info` and `tool_chain`. Focus on the logic, not the format.

**Query 1:** "What was the final score of the game 2015-02-21 - 18-00 Chelsea vs Burnley?"
**Additional Material:** None
* **Analysis (Known Info):** The query is clearly about a specific game.
* **Analysis (Tool Chain):** Must find the game, retrieve its static info (Game Info) and its event history (Match History).
* **Logical Output:**
    * `known_info`: ["$GameContext$"]
    * `tool_chain`: ["game_search", "game_info_retrieval", "game_history_retrieval"]

**Query 2:** "How many goals did the player in this picture score for his senior career?"
**Additional Material:** "image": $["player_image.jpg"]$
* **Analysis (Known Info):** There is an $Image$ and the query is about a $PlayerContext$.
* **Analysis (Tool Chain):** Must identify the player in the image, search for that player's entity information, retrieve specific details from that information.
* **Logical Output:**
    * `known_info`: ["$Image$", "$PlayerContext$"]
    * `tool_chain`: ["entity_recognition", "textual_retrieval_augment"]

**Query 3:** "Who scored the goal in the 2014 World Cup final, and what was the first professional club he ever played for?"
**Additional Material:** None
* **Analysis (Known Info):** The query is about a specific game ($GameContext$) and a specific player ($PlayerContext$).
* **Analysis (Tool Chain):** Must find the game, retrieve its history (to find the goalscorer), then use that player's name to search for their entity information, and retrieve the specific detail (first club).
* **Logical Output:**
    * `known_info`: ["$GameContext$", "$PlayerContext$"]
    * `tool_chain`: ["game_search", "game_history_retrieval", "textual_entity_search", "textual_retrieval_augment"]

## Important Rules
1.  **CRITICAL: Your *only* job is to create a PLAN (Known Info and Tool Chain). Do NOT use your internal, pre-trained knowledge to answer the query. You must create a chain that *finds* all pieces of information using the tools, even if you think you already know the answer.**
2.  You should only use the tools provided in the toolbox to answer the questions and provide the EXACT tool names as listed above.
3.  Use exact item category names with $$ to represent the information categories in the `known_info` field.
4.  Use EXACT tool names WITHOUT any special characters (**, $$, [], etc.) in the `tool_chain` field.
5.  Only respond with the Part 1 analysis (the `known_info` and `tool_chain` values).
6.  Try your best to decompose the question. 

---
---
--- NOW, ANALYZE THE FOLLOWING REQUEST ---

Query: {user_query}
Additional Material: {additional_material}
""")])
    
    return planning_prompt_template

def get_execution_prompt_template() -> ChatPromptTemplate:
    execution_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are a tool execution coordinator for the Soccer Question Answering Assistant."
        ),
        HumanMessagePromptTemplate.from_template(
            """## Task
As a multi-agent core in the Soccer Question Answering Assistant, you are required to execute the following tool chain to answer the question.

**Original User Query:** 
"{user_query}"

**Additional Material:** 
{additional_material}

**Known Information:** 
{known_info}

**Tool Chain to Execute:** 
{tool_chain}
---
## Execution Guidelines
For every time of generation, you should follow the following rules:
- Based on the provided tool chain and execution history, call the next tool. Your task is to generate the exact necessary parameters for the selected tool.
- The requirements and functionality of each tool have been provided. Please rely on that description to generate the correct parameters for the tool.
- Generally, if the tool allows you to rephrase the "query" for better clarity, you should use the execution history and the tool's role to clarify the "query" before making the tool call.
---
## Execution History
The following is all our execution history. You must review this history to inform your next step:
{history}
---
## CRITICAL RULES
1. **DO NOT USE YOUR INTERNAL KNOWLEDGE OR PRE-TRAINED INFORMATION**
2. **You MUST follow the tool chain** and use ONLY the information from the execution history
---
## Next Step
Now, based on all the information above and the execution history, you can start with your call of the next step:
""")])
    return execution_prompt_template