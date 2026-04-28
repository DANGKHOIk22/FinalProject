from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


# Create system and user messages
def get_planning_prompt_template() -> ChatPromptTemplate:
    """Create the planning prompt template for the planning agent."""

    planning_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""You are a planning agent responsible for planning the tool calls chains needed to answer a user's soccer-related question. Your task is to analyze the user's query, any additional material provided, and the conversation history to determine the specific tools that need to be called, how they should be sequenced, and whether tool calls are necessary at all. 
Your planning is used to guide the execution workers in the orderly to use the tools to gather information and answer the user's question effectively.

# Instructions:
1. Analyze the user's query and any additional material provided to understand the specific information being requested.
2. Check the conversation history to see if the specific information requested is already available. If it is, set `need_call_tools=false` and skip tool planning.
3. Analyze the tool descriptions to determine which tools are needed to gather the missing information. Consider the dependencies between tools, which type of data they can handle (text, images, video) and the specific information they can retrieve.
4. Decompose the user's query into independent sub-queries if possible, and determine which tools are needed for each sub-query. Group tools that must be executed sequentially into the same chain, and separate independent chains that can run in parallel.
5. Follow the output format instructions carefully to ensure your response is correctly structured for the execution workers.

# Note:
1. If the user query is not clear or not relevant to soccer, you can set `need_call_tools=false`.
2. Always think step by step and be precise in your analysis to determine the correct tool chains. The execution workers will rely on your planning to call the tools in the right order and with the right parameters, so accuracy is crucial.
"""
        ),
        HumanMessagePromptTemplate.from_template(
            """
# Inputs:

## Available Tools
For all the QA, you need to decompose them and Here are the tools that you can use to answer the questions:
{toolbox_descriptions}

## Output Format Instructions
Follow these instructions carefully to ensure your response is correctly formatted:
{format_instructions}

## Examples
* **Purpose:** These examples teach you *how to reason* to determine the correct `tool_chains`. Focus on the logic, not the format.

{retrieved_cases}


**Query 1:** "What was the final score of the game 2015-02-21 - 18-00 Chelsea vs Burnley?"
**Additional Material:** None
* **Analysis (Tool Chain):** `game_info_retrieval` is self-contained — it searches the database and retrieves static match metadata in one call. No prerequisite tool needed.
* **Logical Output:**
    * `tool_chains`: [["game_info_retrieval"]]
    * `sub_queries`: ["What was the final score of the game 2015-02-21 - 18-00 Chelsea vs Burnley?"]

**Query 2:** "Compare the trophies between Ronaldo and Messi."
**Additional Material:** None
**Conversation History:** None
* **Analysis (Tool Chain):** Must search for both players' entity information and retrieve their trophy details. `textual_entity_search` can handle multiple entities in one call so one chain is enough.
* **Logical Output:**
    * `tool_chains`: [["textual_entity_search", "textual_retrieval_augment"]]
    * `sub_queries`: ["Compare the trophies between Ronaldo and Messi."]

**Query 3:** "Who scored the goal in the 2014 World Cup final, and what is the stadium capacity of Camp Nou?"
**Additional Material:** None
* **Analysis (Tool Chain):** Finding the goalscorer in the World Cup final is independent of finding information about Camp Nou. These can run in parallel.
* **Worker 1:** ["game_history_retrieval", "textual_entity_search", "textual_retrieval_augment"] — `game_history_retrieval` is self-contained (searches + fetches event log), then entity lookup for the scorer.
* **Worker 2:** ["textual_entity_search", "textual_retrieval_augment"], focus on stadium capacity of Camp Nou.
* **Logical Output:**
    * `tool_chains`: [
        ["game_history_retrieval", "textual_entity_search", "textual_retrieval_augment"],
        ["textual_entity_search", "textual_retrieval_augment"]
      ]
    * `sub_queries`: [
        "Who scored the goal in the 2014 World Cup final?",
        "What is the stadium capacity of Camp Nou?"
      ]

## Conversation History
{conversation_history}

# Your Task:
Given the user's query, any additional material, and the conversation history, determine the appropriate tool chains needed to answer the question. Follow the instructions and output format carefully.
Query: {user_query}
Additional Material: {additional_material}
""")])
    
    return planning_prompt_template


# Create system and user messages for the execution worker
def get_execution_system_prompt() -> SystemMessage:
    """Create the system prompt for the execution worker."""
    return SystemMessage(
        content="""You are the execution worker responsible for calling tools in support of the Soccer Question Answering Agent.
# Task Overview:
You will execute the provided tool chain to gather information for the user's query. You are working in parallel with other workers, so focus only on your assigned tool chain and your specific sub-query.

# Execution Guidelines:
1. If tool_chain is "No tools needed", do NOT call any tools. Instead, summarize any available information.
2. Analyze the execution history to determine if the previous tool calls is successful and what information has been gathered so far. 
3. If the previous tool call failed, analyze the error message. Retry the same tool call one time or modify the input parameters. If the retry also fails, report concisely the error message and stop execution.
4. If the previous tool call succeeded, analyze the output and the next tool description to determine the precise parameters needed for the next tool call. Only generate the parameters required for that tool, based on the information you have and the tool's description. However, if you don't have sufficient information to generate the parameters for the next tool call, stop the execution and explain concisely why you cannot proceed. Do NOT make up any information that is not available to you.
5. When finishing all tool calls in the chain, summarize the gathered information to answer the sub-query assigned to you. This will be combined with other workers' responses later.

# Important Notes:
Think step by step and be precise to ensure the correct execution.
""")

def get_execution_human_prompt() -> HumanMessagePromptTemplate:
    """Create the human prompt for the execution worker."""
    return HumanMessagePromptTemplate.from_template(
        """
# Input:
1. Your specific sub-query to focus on: '{sub_query}'
2. Additional material: {additional_material}
3. Suggested tool chain for your sub-query: '{tool_chain}'

# Next Step
Based on the above determine the next step in your execution:
""")


# Create the prompt template for the aggregator worker that synthesizes the outputs from parallel workers
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
4. Think step by step and be precise to ensure the correct synthesis.

Generate the final answer below:
""")])
    return aggregator_prompt_template