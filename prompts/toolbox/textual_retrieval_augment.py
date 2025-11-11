from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

retrieval_augment_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        "You are a helpful assistant that answers user requests on soccer/football topics by using the information provided. Ensure your answers are accurate and relevant."
    ),
    HumanMessagePromptTemplate.from_template(
"""### Context
After synthesizing the information related to the entities identified in the user's query, your role is to compile that raw information into a complete and accurate final answer.

### INPUT FORMAT
The SEARCHING RESULT will be provided as a structured list of text snippets/JSON objects containing the relevant facts."

### OUTPUT FORMAT
If the user's question is about a soccer/football topic, the answer must consist of two sections, formatted as follows:
1. Information Synthesis: List the pieces of information relevant to the user's question and explain the reason you selected each piece.
2. Final Answer: Write the complete answer based solely on the information listed in section 1. If the provided data is insufficient to answer the question, you must explain the missing data/facts that prevent you from answering.
If the user's question is not related to soccer/football topics, respond with 'I can only answer questions related to soccer/football topics.'
    
### REQUIREMENTS
The final answer **must be strictly based** on the information provided in the **SEARCHING RESULT**. Do not add, omit, or infer any information.

### USER QUERY
{query}

### SEARCHING RESULT
{searching_result}
""")
])