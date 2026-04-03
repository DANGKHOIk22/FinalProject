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

Your task involves two sequential parts:
1. Problem Decomposition (Part 1)
- The query you receive has already been clarified (pronouns resolved, abbreviations expanded). Take it as-is.
- Check if the SPECIFIC INFORMATION requested is already available in the conversation history.
- If found in history → set need_call_tools=false. Otherwise, break down the question into independent, parallel tasks and sequential steps.

2. Parallel Tool Application (Part 2)
- Determine which tools can be executed independently in parallel branches.
- Group tools that must be executed sequentially into the same chain.
- Create multiple independent tool chains if there are independent branches of investigation.

## Conversation History
{conversation_history}

## Available Tools
For all the QA, you need to decompose them and Here are the tools that you can use to answer the questions:
{toolbox_descriptions}

## Response Instructions
You must respond with a plan that populates the following fields based on your analysis. The framework will handle formatting.
1.  **tool_chains**: A list of lists of EXACT tool names needed to answer the query. Each inner list represents an independent chain of tools that can run in parallel. Tools within an inner list run sequentially.
2.  **sub_queries**: A list of strings, corresponding to each tool chain in `tool_chains`. Each string should be the specific decomposed part of the user query that the respective tool chain is responsible for answering.

## Output Format Instructions
Follow these instructions carefully to ensure your response is correctly formatted:
{format_instructions}

## Examples
* **Purpose:** These examples teach you *how to reason* to determine the correct `tool_chains`. Focus on the logic, not the format.

**Query 1:** "What was the final score of the game 2015-02-21 - 18-00 Chelsea vs Burnley?"
**Additional Material:** None
* **Analysis (Tool Chain):** Must find the game, retrieve its static info (Game Info) and its event history (Match History). These must be done sequentially as they depend on the same game.
* **Logical Output:**
    * `tool_chains`: [["game_search", "game_info_retrieval", "game_history_retrieval"]]
    * `sub_queries`: ["What was the final score of the game 2015-02-21 - 18-00 Chelsea vs Burnley?"]

**Query 2:** "Compare the trophies between Ronaldo and Messi."
**Additional Material:** None
**Conversation History:** None
* **Analysis (Tool Chain):** Must search for both players' entity information and retrieve their trophy details. This can be done in parallel for each player. Wait, the textual_entity_search can handle multiple entities in one call so doing it sequentially or in one chain is fine. However, if handled separately:
* Let's say we want to do it in one chain to save calls because the tool supports multiple entities:
* **Logical Output:**
    * `tool_chains`: [["textual_entity_search", "textual_retrieval_augment"]]
    * `sub_queries`: ["Compare the trophies between Ronaldo and Messi."]

**Query 3:** "Who scored the goal in the 2014 World Cup final, and what is the stadium capacity of Camp Nou?"
**Additional Material:** None
* **Analysis (Tool Chain):** Finding the goalscorer in the World Cup final is independent of finding information about Camp Nou. These can run in parallel. 
* **Worker 1:** ["game_search", "game_history_retrieval", "textual_entity_search", "textual_retrieval_augment"], focus on goalscorer in 2014 World Cup final.
* **Worker 2:** ["textual_entity_search", "textual_retrieval_augment"], focus on stadium capacity of Camp Nou.
* **Logical Output:**
    * `tool_chains`: [
        ["game_search", "game_history_retrieval", "textual_entity_search", "textual_retrieval_augment"],
        ["textual_entity_search", "textual_retrieval_augment"]
      ]
    * `sub_queries`: [
        "Who scored the goal in the 2014 World Cup final?",
        "What is the stadium capacity of Camp Nou?"
      ]
    
## Important Rules
1.  **CRITICAL: Your *only* job is to create a PLAN. Do NOT use your internal, pre-trained knowledge to answer the query. You must create chains that *find* all pieces of information using the tools, even if you think you already know the answer.**
2.  **CRITICAL: Check conversation history FIRST.** Only skip tools (need_call_tools=false) if the SPECIFIC INFORMATION requested is already available in the conversation history. If an entity is mentioned but the specific information requested (e.g., goals, trophies, clubs) is NOT there, you MUST include tools to retrieve that missing information.
3.  You should only use the tools provided in the toolbox to answer the questions and provide the EXACT tool names as listed above.
4.  Should use segment tool first if the question involves an image to identify the entity more accurately.
5.  Route the request based on input type: Use entity_recognition for image analysis OR textual_entity_search for text analysis. Never use both sequentially for the same entity.
6.  Try your best to decompose the question into independent parallel tasks when appropriate.

---
---
--- NOW, ANALYZE THE FOLLOWING REQUEST ---

Query: {user_query}
Additional Material: {additional_material}
""")])
    
    return planning_prompt_template


def get_execution_prompt_template() -> ChatPromptTemplate:
    """Create the execution prompt template for a single execution worker."""

    execution_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are the execution worker responsible for calling tools in support of the Soccer Question Answering Agent."
        ),
        HumanMessagePromptTemplate.from_template(
            """# Task Overview:
You will execute the provided tool chain to gather information for the user's query. You are working in parallel with other workers, so focus only on your assigned tool chain and your specific sub-query.

# Execution Guidelines:
1. If tool_chain is "No tools needed", do NOT call any tools. Instead, summarize any available information.
2. Analyze the execution history to determine if the previous tool calls is successful and what information has been gathered so far. 
3. If the previous tool call failed, analyze the error message. Retry the same tool call one time or modify the input parameters. If the retry also fails, report concisely the error message and stop execution.
4. If the previous tool call succeeded, analyze the output and the next tool description to determine the precise parameters needed for the next tool call. Only generate the parameters required for that tool, based on the information you have and the tool's description. However, if you don't have sufficient information to generate the parameters for the next tool call, stop the execution and explain concisely why you cannot proceed. Do NOT make up any information that is not available to you.
5. When finishing all tool calls in the chain, summarize the gathered information to answer the sub-query assigned to you. This will be combined with other workers' responses later.

# Input:
1. Your specific sub-query to focus on: "{sub_query}"
2. Additional material: {additional_material}
3. Suggested tool chain for your sub-query: {tool_chain}
4. Execution history of your tool chain:
{history}

# Next Step
Based on the above determine the next step in your execution:
""")])
    return execution_prompt_template

def get_aggregator_prompt_template() -> ChatPromptTemplate:
    """Create the aggregator prompt template that synthesized worker outputs."""
    
    aggregator_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are the synthesis agent responsible for combining findings from multiple parallel tasks to answer a user's query."
        ),
        HumanMessagePromptTemplate.from_template(
            """# Task Overview:
You need to provide the final definitive answer to the user's query based on the aggregated findings from independent parallel workers.

**Original user query:**
"{user_query}"

**Additional material:**
{additional_material}

**Conversation history:**
{conversation_history}

# Worker Findings:
Below are the summarized findings from each parallel worker that investigated the query.
{worker_results}

# Critical Rules
1. Integrate all findings to fully address all parts of the user's query.
2. If the workers encountered errors or could not find the information, state what is known.
3. Base your final response ONLY on the provided worker findings and conversation history, without making up facts.
4. Provide a coherent, polite, natural language response.

Generate the final answer below:
""")])
    return aggregator_prompt_template