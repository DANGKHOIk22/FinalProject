from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

def get_choice_selection_prompt_template() -> ChatPromptTemplate:
    """Create a prompt template for selecting the most appropriate answer choice from a set of closed-ended options. It is used in the 'choice_selection' tool."""

    choice_selection_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a helpful assistant that selects the most appropriate answer choice from a set of closed-ended (multiple-choice) options based on an open-ended answer."
        ),
        HumanMessagePromptTemplate.from_template(
        """
        # Instruction: 
        Given the following query containing a question and its options, along with an open-ended answer, identify the most appropriate answer choice that best corresponds to the open-ended answer.
        # Input Format:
        Query: {query}
        Open-ended Answer: {open_ended_answer}
        # Output: 
        Provide the selected choice in the format 'o1', 'o2', etc., corresponding to the options in the query. Only output the choice without any additional text.
        # Task: 
        Based on the provided query and open-ended answer, determine which option is the most suitable choice. Ensure that your selection aligns closely with the content and intent of the open-ended answer.
        """
        )
    ])
    return choice_selection_prompt_template