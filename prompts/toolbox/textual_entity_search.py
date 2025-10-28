from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Prompt template for entity extraction
# This ChatPromptTemplate guides the LLM to perform NER on soccer-related queries.
extract_entity_prompt = ChatPromptTemplate.from_messages([
      SystemMessage(
          "You are a helpful assistant that extracts soccer-related entities from user queries. "
          "Extract only entities that are clearly mentioned in the query. "
          "Be precise and extract the exact names as they appear."
      ),
          HumanMessagePromptTemplate.from_template(
                """### TASK: Your task is to analyze the # INPUT and perform Named Entity Recognition (NER). You must:
    1. Identify the specific entity mentioned.
    2. Determine the correct ENTITY TYPE based on the context provided in the # INPUT.
    3. Extract the exact name of the entity.
    ### CONTEXT AND AMBIGUITY RESOLUTION:
    When a proper noun is mentioned, its entity type can often be ambiguous. Therefore, you must use the full context of the # INPUT to infer the correct entity type. Crucially, if the context in the # INPUT is insufficient to confidently classify the entity, or if you are uncertain, you must classify the entity as 'unknown'. Prioritize 'unknown' in cases of ambiguity or low confidence.**
    # ENTITY TYPE
    The defined entity types are: **player**, **referee**, **team**, **venue**, **unknown**.
    Note: If the entity is a **coach**, it must be classified as a **player**.
    ### OUTPUT FORMAT
    The extracted entity name must exactly match its appearance in the question.    
    {output_format}
    ### INPUT: {question}
    ### EXAMPLES
    Example 1: Ambiguity leading to 'unknown'
    INPUT: How many goals did Messi score?
    OUTPUT:
    {{
    "unknown": ["Messi"],
    "player": null,
    "team": null,
    "venue": null,
    "referee": null
    }}
    
    Example 2: Clear context and multiple entities 
    INPUT: What is the salary of Messi player in Paris Saint-Germain?
    OUTPUT:
    {{
    "unknown": null,
    "player": ["Messi"],
    "team": ["Paris Saint-Germain"],
    "venue": null,
    "referee": null
    }}
    
    Example 4:
    INPUT: Which referee officiated the match at Wembley Stadium?
    OUTPUT:
    {{
    "unknown": null,
    "player": null,
    "team": null,
    "venue": ["Wembley Stadium"],
    "referee": null
    }}
    
    Example 5: Highly ambiguous entity
    INPUT: When did Zidane start his career?
    OUTPUT:
    {{
    "unknown": ["Zidane"],
    "player": null,
    "team": null,
    "venue": null,
    "referee": null
    }}
    """),
    ])