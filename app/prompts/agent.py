from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


# Create system and user messages
def get_planning_prompt_template() -> ChatPromptTemplate:
    """Create the planning prompt template for the planning agent."""

    planning_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are a multi-modal agent that can answer questions about soccer knowledge."
        ),
        HumanMessagePromptTemplate.from_template(
            """For each question, you will receive:
- A question about soccer considering different aspects of soccer
- You might also receive one or more video clips or images as context
- You might also receive conversation history showing previous interactions

Your task involves three sequential parts:
1. Problem Decomposition (Part 1)
- **CRITICAL: If the query uses pronouns (he, him, she, her, his, they, them, it, etc.) instead of specific entity names, you MUST check the conversation history FIRST to identify which specific entity is being referenced.**
- After resolving any pronouns to specific entities, check if the SPECIFIC INFORMATION requested about that entity is already available in the conversation history
- If the entity is mentioned in conversation history BUT the specific information requested is NOT there, you MUST include tools to retrieve that information
- Identify available information from the query and context
- Break down the question into sequential steps
2. Sequential Tool Application (Part 2)
- Execute one tool at a time
- Record each tool's output
- Continue until sufficient information is gathered
3. Solution Synthesis (Part 3)
- Integrate all results
- Generate final answer

## Conversation History
{conversation_history}

## Available Tools
For all the QA, you need to decompose them and Here are the tools that you can use to answer the questions:
{toolbox_descriptions}

## Response Instructions
You must respond with a plan that populates the following two fields based on your analysis. The framework will handle formatting.
1.  **known_info**: A list of information categories explicitly mentioned in the query and material (e.g., $GameContext$, $PlayerContext$, $Image$, $ConversationHistory$).
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

**Query 4:** "How many goals did the player on the left side of the image, wearing a white jersey, score in his senior career?"
**Additional Material:** "image": $["player_image.jpg"]$
* **Analysis (Known Info):** There is an $Image$ and the query is about a $PlayerContext$.
* **Analysis (Tool Chain):** Must identify the player in the image, search for that player's entity information, retrieve specific details from that information.
* **Logical Output:**
    * `known_info`: ["$Image$", "$PlayerContext$"]
    * `tool_chain`: ["segment", "entity_recognition", "textual_retrieval_augment"]

**Query 5:** "Compare the trophies between Ronaldo and Messi."
**Additional Material:** None
**Conversation History:** None
* **Analysis (Known Info):** The query involves multiple players ($PlayerContext$).
* **Analysis (Tool Chain):** Must search for both players' entity information and retrieve their trophy details. Tools can handle multiple entities, so NO repetition is needed.
* **Logical Output:**
    * `known_info`: ["$PlayerContext$"]
    * `tool_chain`: ["textual_entity_search", "textual_retrieval_augment"]

**Query 6:** "Compare his goals with Messi."
**Additional Material:** None
**Conversation History:** 
- User: "How many goals did Ronaldo score during his career?"
- Assistant: "Ronaldo scored 903 goals during his senior career."
* **Analysis (Known Info):** "His" refers to Ronaldo from conversation history. Ronaldo's goal count is already known from $ConversationHistory$. Only Messi's information needs to be retrieved.
* **Analysis (Tool Chain):** Only need to search for Messi's entity information and retrieve his goals. NO need to search for Ronaldo again since that information is in conversation history.
* **Logical Output:**
    * `known_info`: ["$PlayerContext$", "$ConversationHistory$"]
    * `tool_chain`: ["textual_entity_search", "textual_retrieval_augment"]
    
## Important Rules
1.  **CRITICAL: Your *only* job is to create a PLAN (Known Info and Tool Chain). Do NOT use your internal, pre-trained knowledge to answer the query. You must create a chain that *finds* all pieces of information using the tools, even if you think you already know the answer.**
2.  **CRITICAL: Check conversation history FIRST.** Only skip tools if the SPECIFIC INFORMATION requested is already available in the conversation history. If an entity is mentioned but the specific information requested (e.g., goals, trophies, clubs) is NOT there, you MUST include tools to retrieve that missing information.
3.  **CRITICAL: Pronoun Resolution.** If the query uses pronouns (he, him, she, her, his, they, them, it, etc.) without naming a specific entity, you MUST:
   a) Examine the conversation history to determine which specific entity is being referenced
   b) Check if the SPECIFIC INFORMATION requested about that entity is already in conversation history
   c) If the information is missing, include tools to retrieve it even though the entity was previously mentioned
4.  You should only use the tools provided in the toolbox to answer the questions and provide the EXACT tool names as listed above.
5.  Use exact item category names with $$ to represent the information categories in the `known_info` field.
6.  Use EXACT tool names WITHOUT any special characters (**, $$, [], etc.) in the `tool_chain` field.
7.  Only respond with the Part 1 analysis (the `known_info` and `tool_chain` values).
8. Should use segement tool first if the question involves an image to identify the entity more accurately.
9. Route the request based on input type: Use entity_recognition for image analysis OR textual_entity_search for text analysis. Never use both sequentially for the same entity.
10. **CRITICAL: Do NOT repeat tools for multiple entities.** Each tool can handle multiple entities in a single call. Even if the query involves comparing or analyzing multiple players, teams, or games, use each tool only ONCE in the chain.
11.  Try your best to decompose the question. 

---
---
--- NOW, ANALYZE THE FOLLOWING REQUEST ---

Query: {user_query}
Additional Material: {additional_material}
""")])
    
    return planning_prompt_template


def get_execution_prompt_template() -> ChatPromptTemplate:
    """Create the execution prompt template for the execution agent."""

    execution_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are the execution coordinator responsible for calling tools in support of the Soccer Question Answering Agent."
        ),
        HumanMessagePromptTemplate.from_template(
            """# Task Overview:
You will execute the provided tool chain to answer the user's query by producing the next tool call and its exact parameters.

**Original user query:**
"{user_query}"

**Additional material:**
{additional_material}

**Conversation history (use only when giving the final answer after all tools executed):**
{conversation_history}

**Known information:**
{known_info}

**Tool chain to execute:**
{tool_chain}

# Execution Guidelines:
**CRITICAL: If tool_chain is "No tools needed", do NOT call any tools. Instead, directly provide the final answer using available information.**

For every time of generation, you should follow the following rules:
- At each step, select the next tool in the chain and generate only the precise parameters required for that tool call.
- If a tool call fails, retry it one time. If the retry also fails, report the error message and stop execution.
- Use the tool descriptions to determine required parameters. When permitted, refine the tool's query using the execution history and the tool's stated role to improve clarity.
- Rely only on information available in the execution history and the provided materials; do not use internal or pre-trained knowledge.

# Execution History:
Review the complete execution history below to inform your next action:
{history}

# Critical Rules
1. **CRITICAL: If tool_chain is "No tools needed", NEVER call any tools. Provide the final answer directly.**
2. Do not use internal knowledge or pre-trained information.
3. Follow the tool chain exactly and use only information from the execution history.
4. Do not skip any tool in the chain.

# Next Step
Based on the context and execution history, decide whether another tool call is required. If so, output the exact tool invocation with all necessary parameters. If not (all tools completed OR tool_chain is "No tools needed"), end execution with a clear, polite response to the user summarizing the gathered information (you may reference conversation history at this point), without calling further tools.
""")])
    return execution_prompt_template